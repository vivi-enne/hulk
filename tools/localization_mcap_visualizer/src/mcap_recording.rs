use std::{
    collections::BTreeMap,
    fmt, fs,
    path::Path,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use booster::ImuState;
use color_eyre::{Result, eyre::Context as _};
use coordinate_systems::{Field, Pixel, Robot};
use enumset::enum_set;
use field_mark_association::{FieldMarkAssociations, GlobalLocalizationDebug};
use image::RgbImage;
use kinematics::robot_kinematics::RobotKinematics;
use linear_algebra::{IntoTransform, Isometry3, vector};
use localization_3d::SolveDiagnostics;
use mcap::{Message, MessageStream, read::Options};
use projection::{camera_matrix::CameraMatrix, intrinsic::Intrinsic};
use ros_z::time::Time;
use ros_z_cdr::{LittleEndian, from_bytes};
use ros2::sensor_msgs::image::Image as RosImage;
use serde::{Deserialize, de::DeserializeOwned};
use types::{
    field_dimensions::FieldDimensions,
    object_detection::{Object, RobocupObjectLabel},
    stereo_camera_info::StereoCameraInfo,
    stereo_image_pair::StereoImagePair,
    time_wrapper::TimeWrapper,
    visual_odometry::VisualOdometryDelta,
};

use crate::nearest_by_distance;
use crate::replay::TimestampMode;

pub const TOPIC_IMU_STATE: &str = "inputs/imu_state";
pub const TOPIC_STEREO_IMAGE_PAIR: &str = "inputs/stereo_image_pair";
pub const TOPIC_STEREO_CAMERA_INFO: &str = "inputs/stereo_camera_info";
pub const TOPIC_ROBOT_KINEMATICS: &str = "robot_kinematics";
pub const TOPIC_CAMERA_MATRIX: &str = "camera_matrix";
pub const TOPIC_DETECTED_OBJECTS: &str = "detected_objects";
pub const TOPIC_DETECTED_OBJECTS_ANNOUNCE: &str = "detected_objects/announce";
pub const TOPIC_FIELD_MARK_ASSOCIATIONS: &str = "field_mark_association/associations";
pub const TOPIC_FIELD_DIMENSIONS: &str = "field_dimensions";
pub const TOPIC_LOCALIZATION: &str = "localization";
pub const TOPIC_VISUAL_ODOMETRY: &str =
    "visual_odometry/current_left_camera_to_previous_left_camera";
pub const TOPIC_CALIBRATED_INTRINSICS: &str = "debug/calibrated_intrinsics";
pub const TOPIC_GLOBAL_LOCALIZATION_DEBUG: &str = "debug/global_localization";
pub const TOPIC_SOLVE_DIAGNOSTICS: &str = "debug/solve_diagnostics";

pub const SNAPSHOT_MAX_TIME_DISTANCE: Duration = Duration::from_millis(100);
pub const TRAJECTORY_MAX_SAMPLE_GAP_SECONDS: f64 = 0.5;

pub struct Recording {
    events: Vec<RecordedEvent>,
    images: Vec<StereoImageIndex>,
    images_by_embedded_time: Vec<usize>,
    detected_objects_by_image: Vec<Option<usize>>,
    detected_objects_image_times: BTreeMap<i64, DetectedObjectsImageTime>,
    snapshot_index: SnapshotIndex,
    pub first_camera_matrix: CameraMatrix,
    pub stereo_camera_info: Option<StereoCameraInfo>,
    pub field_dimensions: Option<FieldDimensions>,
    topic_counts: BTreeMap<String, usize>,
}

impl Recording {
    pub fn load(path: &Path) -> Result<Self> {
        let bytes: Arc<[u8]> = fs::read(path)
            .wrap_err_with(|| format!("failed to read {}", path.display()))?
            .into();
        let mut events = Vec::new();
        let mut images = Vec::new();
        let mut first_camera_matrix = None;
        let mut stereo_camera_info = None;
        let mut field_dimensions = None;
        let mut topic_counts = BTreeMap::new();
        let mut detected_object_announcements = BTreeMap::new();
        let mut field_mark_associations = Vec::new();

        for (order, message) in
            MessageStream::new_with_options(&bytes, enum_set!(Options::IgnoreEndMagic))
                .wrap_err("failed to open MCAP message stream")?
                .enumerate()
        {
            let message = match message {
                Ok(message) => message,
                Err(error) => {
                    tracing::warn!(
                        ?error,
                        recovered_messages = order,
                        "stopped reading MCAP after damaged or incomplete record"
                    );
                    break;
                }
            };
            *topic_counts
                .entry(message.channel.topic.clone())
                .or_default() += 1;

            let log_time = system_time_from_nanos(message.log_time);
            let publish_time = system_time_from_nanos(message.publish_time);
            let sequence = i64::from(message.sequence);
            let kind = match message.channel.topic.as_str() {
                TOPIC_IMU_STATE => Some(EventKind::Imu(decode_recorded_message(&message)?)),
                TOPIC_STEREO_CAMERA_INFO => {
                    stereo_camera_info = Some(decode_recorded_message(&message)?);
                    None
                }
                TOPIC_VISUAL_ODOMETRY => Some(EventKind::VisualOdometry(
                    decode_recorded_visual_odometry(&message)?,
                )),
                TOPIC_ROBOT_KINEMATICS => Some(EventKind::RobotKinematics(Box::new(
                    decode_recorded_message(&message)?,
                ))),
                TOPIC_CAMERA_MATRIX => {
                    let camera_matrix = decode_recorded_camera_matrix(&message)?;
                    if first_camera_matrix.is_none() {
                        first_camera_matrix = Some(camera_matrix.inner.clone());
                    }
                    Some(EventKind::CameraMatrix(camera_matrix))
                }
                TOPIC_DETECTED_OBJECTS => Some(EventKind::DetectedObjects(DetectedObjectsFrame {
                    objects: decode_recorded_message(&message)?,
                    image_time: None,
                    publish_time,
                    announcement_log_time: None,
                    sequence_number: sequence,
                })),
                TOPIC_DETECTED_OBJECTS_ANNOUNCE => {
                    let announcement: WireAnnouncement = decode_recorded_message(&message)?;
                    detected_object_announcements
                        .insert(announcement.sequence_number, (announcement.time, log_time));
                    None
                }
                TOPIC_FIELD_MARK_ASSOCIATIONS => {
                    let source_time = decode_time_prefix(&message.data).wrap_err_with(|| {
                        format!(
                            "failed to decode {TOPIC_FIELD_MARK_ASSOCIATIONS} time prefix order {order} sequence {}",
                            message.sequence
                        )
                    })?;
                    field_mark_associations.push(FieldMarkAssociationTiming {
                        order,
                        log_time,
                        source_time,
                    });
                    Some(EventKind::FieldMarkAssociations(decode_recorded_message(
                        &message,
                    )?))
                }
                TOPIC_FIELD_DIMENSIONS => {
                    let dimensions = decode_recorded_message(&message)?;
                    field_dimensions = Some(dimensions);
                    None
                }
                TOPIC_LOCALIZATION => Some(EventKind::RecordedLocalization(
                    decode_recorded_message(&message)?,
                )),
                TOPIC_CALIBRATED_INTRINSICS => Some(EventKind::CalibratedIntrinsics(
                    decode_recorded_message(&message)?,
                )),
                TOPIC_GLOBAL_LOCALIZATION_DEBUG => Some(EventKind::GlobalLocalizationDebug(
                    decode_recorded_message(&message)?,
                )),
                TOPIC_SOLVE_DIAGNOSTICS => Some(EventKind::SolveDiagnostics(
                    decode_recorded_message(&message)?,
                )),
                TOPIC_STEREO_IMAGE_PAIR => {
                    let data: Arc<[u8]> = message.data.into_owned().into();
                    let embedded_time = decode_time_prefix(&data).wrap_err_with(|| {
                        format!(
                            "failed to decode {TOPIC_STEREO_IMAGE_PAIR} time prefix order {order} sequence {}",
                            message.sequence
                        )
                    })?;
                    images.push(StereoImageIndex {
                        order,
                        log_time,
                        publish_time,
                        embedded_time,
                        data,
                    });
                    None
                }
                _ => None,
            };

            if let Some(kind) = kind {
                events.push(RecordedEvent {
                    order,
                    log_time,
                    publish_time,
                    kind,
                });
            }
        }

        events.sort_by_key(|event| (nanos_since_epoch(event.log_time), event.order));
        images.sort_by_key(|image| (image.embedded_time.as_nanos(), image.order));
        field_mark_associations.sort_by_key(|association| {
            (nanos_since_epoch(association.log_time), association.order)
        });
        let mut images_by_embedded_time = (0..images.len()).collect::<Vec<_>>();
        images_by_embedded_time
            .sort_by_key(|&index| (images[index].embedded_time.as_nanos(), images[index].order));
        let mut detected_objects_image_sources = detected_object_announcements;
        for (sequence, time) in index_detected_objects_sources_from_field_mark_associations(
            &events,
            &field_mark_associations,
        ) {
            detected_objects_image_sources
                .entry(sequence)
                .or_insert(time);
        }
        let detected_objects_by_image = index_detected_objects_by_image(
            &events,
            &images,
            &images_by_embedded_time,
            &detected_objects_image_sources,
        );
        let detected_objects_image_times = index_detected_objects_image_times(
            &images,
            &images_by_embedded_time,
            &detected_objects_image_sources,
        );
        for event in &mut events {
            if let EventKind::DetectedObjects(frame) = &mut event.kind
                && let Some((image_time, announcement_log_time)) =
                    detected_objects_image_sources.get(&frame.sequence_number)
            {
                frame.image_time = Some(*image_time);
                frame.announcement_log_time = Some(*announcement_log_time);
            }
        }
        let snapshot_index = SnapshotIndex::new(&events);

        Ok(Self {
            events,
            images,
            images_by_embedded_time,
            detected_objects_by_image,
            detected_objects_image_times,
            snapshot_index,
            first_camera_matrix: first_camera_matrix
                .ok_or_else(|| color_eyre::eyre::eyre!("recording has no camera_matrix topic"))?,
            stereo_camera_info,
            field_dimensions,
            topic_counts,
        })
    }

    pub fn start_log_time(&self) -> SystemTime {
        self.events
            .iter()
            .map(|event| event.log_time)
            .chain(self.images.iter().map(|image| image.log_time))
            .min()
            .unwrap_or(UNIX_EPOCH)
    }

    pub fn start_source_time(&self) -> SystemTime {
        self.events
            .iter()
            .map(|event| event.publish_time)
            .chain(self.images.iter().map(|image| image.publish_time))
            .min()
            .unwrap_or(UNIX_EPOCH)
    }

    pub fn graph_start_time(&self, timestamp_mode: TimestampMode) -> SystemTime {
        match timestamp_mode {
            TimestampMode::McapPublish => self.start_source_time(),
            TimestampMode::Embedded => self.start_display_time(),
        }
    }

    pub fn end_log_time(&self) -> SystemTime {
        self.events
            .iter()
            .map(|event| event.log_time)
            .chain(self.images.iter().map(|image| image.log_time))
            .max()
            .unwrap_or_else(|| self.start_log_time())
    }

    pub fn start_display_time(&self) -> SystemTime {
        self.events
            .iter()
            .map(RecordedEvent::display_time)
            .chain(self.images.iter().map(StereoImageIndex::display_time))
            .min()
            .unwrap_or(UNIX_EPOCH)
    }

    pub fn end_display_time(&self) -> SystemTime {
        self.events
            .iter()
            .map(RecordedEvent::display_time)
            .chain(self.images.iter().map(StereoImageIndex::display_time))
            .max()
            .unwrap_or_else(|| self.start_display_time())
    }

    pub fn duration(&self) -> Duration {
        match self
            .end_display_time()
            .duration_since(self.start_display_time())
        {
            Ok(duration) => duration,
            Err(_) => Duration::ZERO,
        }
    }

    pub fn events(&self) -> &[RecordedEvent] {
        &self.events
    }

    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    pub fn image_count(&self) -> usize {
        self.images.len()
    }

    pub fn image_id_from_index(&self, index: usize) -> Option<StereoImageId> {
        if index < self.images.len() {
            Some(StereoImageId(index))
        } else {
            None
        }
    }

    pub fn image_log_time(&self, image_id: StereoImageId) -> Option<SystemTime> {
        self.images
            .get(image_id.index())
            .map(|image| image.log_time)
    }

    pub fn image_display_time(&self, image_id: StereoImageId) -> Option<SystemTime> {
        self.images
            .get(image_id.index())
            .map(StereoImageIndex::display_time)
    }

    pub fn topic_count(&self) -> usize {
        self.topic_counts.len()
    }

    pub fn display_time_at_seconds(&self, seconds: f64) -> SystemTime {
        self.start_display_time() + Duration::from_secs_f64(seconds.max(0.0))
    }

    pub fn topic_message_count(&self, topic: &str) -> usize {
        self.topic_counts.get(topic).copied().unwrap_or_default()
    }

    pub fn log_time_at_seconds(&self, seconds: f64) -> SystemTime {
        self.start_log_time() + Duration::from_secs_f64(seconds.max(0.0))
    }

    pub fn seconds_since_start(&self, time: SystemTime) -> f64 {
        self.seconds_since_display_start(time)
    }

    pub fn seconds_since_log_start(&self, time: SystemTime) -> f64 {
        seconds_since(time, self.start_log_time())
    }

    pub fn seconds_since_display_start(&self, time: SystemTime) -> f64 {
        seconds_since(time, self.start_display_time())
    }

    pub fn aligned_image_time(&self, embedded_time: Time) -> Option<SystemTime> {
        let target_nanos = embedded_time.as_nanos();
        let next = self
            .images_by_embedded_time
            .partition_point(|&index| self.images[index].embedded_time.as_nanos() <= target_nanos);
        nearest_by_distance(
            next.checked_sub(1)
                .and_then(|index| self.images_by_embedded_time.get(index))
                .and_then(|&index| self.images.get(index))
                .map(|image| {
                    (
                        image,
                        u128::from(image.embedded_time.as_nanos().abs_diff(target_nanos)),
                    )
                }),
            self.images_by_embedded_time
                .get(next)
                .and_then(|&index| self.images.get(index))
                .map(|image| {
                    (
                        image,
                        u128::from(image.embedded_time.as_nanos().abs_diff(target_nanos)),
                    )
                }),
        )
        .filter(|candidate| {
            Duration::from_nanos(
                u128::from(candidate.embedded_time.as_nanos().abs_diff(target_nanos))
                    .min(u64::MAX as u128) as u64,
            ) <= SNAPSHOT_MAX_TIME_DISTANCE
        })
        .map(|candidate| candidate.publish_time)
    }

    pub fn detected_objects_image_publish_time(&self, event: &RecordedEvent) -> Option<SystemTime> {
        let EventKind::DetectedObjects(frame) = &event.kind else {
            return None;
        };
        self.detected_objects_image_times
            .get(&frame.sequence_number)
            .map(|time| time.publish_time)
    }

    pub fn latest_snapshot(&self, display_time: SystemTime) -> RecordingSnapshot {
        let image_id = self.nearest_image_id(display_time);
        let image_display_time = image_id
            .and_then(|image_id| self.images.get(image_id.index()))
            .map(StereoImageIndex::display_time);
        let visual_time = image_display_time.unwrap_or(display_time);
        let mut snapshot = RecordingSnapshot {
            image_id,
            image_display_time,
            ..Default::default()
        };

        if let Some(EventKind::CameraMatrix(camera_matrix)) = self
            .snapshot_index
            .nearest_camera_matrix(&self.events, visual_time)
            .map(|event| &event.kind)
        {
            snapshot.camera_matrix = Some(camera_matrix.clone());
        }
        let detected_objects = if let Some(image_id) = image_id {
            self.detected_objects_for_image(image_id)
        } else {
            self.snapshot_index
                .nearest_detected_objects(&self.events, visual_time)
                .and_then(|event| match &event.kind {
                    EventKind::DetectedObjects(frame) => Some((frame, frame.display_time())),
                    _ => None,
                })
        };
        if let Some((frame, time)) = detected_objects {
            snapshot.detected_objects = frame.objects.clone();
            snapshot.detected_objects_time = Some(time);
            snapshot.detected_objects_frame = Some(frame.clone());
        }
        if let Some(EventKind::RecordedLocalization(localization)) = self
            .snapshot_index
            .latest_recorded_localization(&self.events, display_time)
            .map(|event| &event.kind)
        {
            snapshot.recorded_localization = *localization;
        }
        if let Some(EventKind::RobotKinematics(robot_kinematics)) = self
            .snapshot_index
            .nearest_robot_kinematics(&self.events, visual_time)
            .map(|event| &event.kind)
        {
            snapshot.robot_kinematics = Some(TimeWrapper {
                time: robot_kinematics.time,
                inner: Arc::new(robot_kinematics.inner.clone()),
            });
        }
        if let Some(event) = self
            .snapshot_index
            .latest_calibrated_intrinsics(&self.events, visual_time)
            && let EventKind::CalibratedIntrinsics(intrinsics) = &event.kind
        {
            snapshot.calibrated_intrinsics = Some(*intrinsics);
            snapshot.calibrated_intrinsics_time = Some(event.publish_time);
        }
        if let Some(event) = self
            .snapshot_index
            .nearest_field_mark_associations(&self.events, visual_time)
            && let EventKind::FieldMarkAssociations(associations) = &event.kind
        {
            snapshot.field_mark_associations = Some(associations.clone());
        }
        if let Some(event) = self
            .snapshot_index
            .nearest_global_localization_debug(&self.events, visual_time)
            && let EventKind::GlobalLocalizationDebug(debug) = &event.kind
        {
            snapshot.global_localization_debug = debug.clone();
            snapshot.global_localization_debug_time = Some(event.display_time());
        }
        if let Some(event) = self
            .snapshot_index
            .latest_solve_diagnostics(&self.events, display_time)
            && let EventKind::SolveDiagnostics(diagnostics) = &event.kind
        {
            snapshot.solve_diagnostics = Some(diagnostics.clone());
        }

        snapshot
    }

    pub fn recorded_localization_trajectory(&self) -> Vec<TrajectoryPoint> {
        let mut trajectory = Vec::new();
        let mut segment_id = 0;
        let mut last_was_none = false;

        for event in &self.events {
            let EventKind::RecordedLocalization(localization) = &event.kind else {
                continue;
            };
            match localization {
                Some(field_to_robot) => {
                    if last_was_none {
                        segment_id += 1;
                        last_was_none = false;
                    }
                    trajectory.push(TrajectoryPoint {
                        seconds: self.seconds_since_display_start(event.display_time()),
                        robot_to_field: field_to_robot.inverse().inner.cast().framed_transform(),
                        segment_id,
                    });
                }
                None => {
                    last_was_none = true;
                }
            }
        }

        trajectory
    }

    pub fn decode_stereo_image(&self, image_id: StereoImageId) -> Result<StereoFrame> {
        let Some(index) = self.images.get(image_id.index()) else {
            return Err(color_eyre::eyre::eyre!(
                "image id {image_id} is out of bounds"
            ));
        };

        let stereo: TimeWrapper<StereoImagePair> =
            decode_message(&index.data).wrap_err_with(|| {
                format!(
                    "failed to decode {TOPIC_STEREO_IMAGE_PAIR} image id {image_id} order {}",
                    index.order
                )
            })?;
        Ok(StereoFrame {
            sequence: index.order as u64,
            source_time: stereo.time,
            log_time: index.log_time,
            publish_time: index.publish_time,
            left: camera_image_from_ros(stereo.inner.left)?,
            right: camera_image_from_ros(stereo.inner.right)?,
        })
    }

    pub fn decode_stereo_pair(
        &self,
        image_id: StereoImageId,
    ) -> Result<TimeWrapper<StereoImagePair>> {
        let Some(index) = self.images.get(image_id.index()) else {
            return Err(color_eyre::eyre::eyre!(
                "image id {image_id} is out of bounds"
            ));
        };

        decode_message(&index.data).wrap_err_with(|| {
            format!(
                "failed to decode {TOPIC_STEREO_IMAGE_PAIR} image id {image_id} order {}",
                index.order
            )
        })
    }

    fn nearest_image_id(&self, display_time: SystemTime) -> Option<StereoImageId> {
        if self.images.is_empty() {
            return None;
        }
        let next = self
            .images
            .partition_point(|image| image.display_time() <= display_time);
        nearest_by_distance(
            next.checked_sub(1).map(|previous| {
                (
                    previous,
                    nanos_abs_diff(self.images[previous].display_time(), display_time),
                )
            }),
            self.images.get(next).map(|next_image| {
                (
                    next,
                    nanos_abs_diff(next_image.display_time(), display_time),
                )
            }),
        )
        .filter(|&index| {
            abs_duration(self.images[index].display_time(), display_time)
                <= SNAPSHOT_MAX_TIME_DISTANCE
        })
        .map(StereoImageId)
    }

    fn detected_objects_for_image(
        &self,
        image_id: StereoImageId,
    ) -> Option<(&DetectedObjectsFrame, SystemTime)> {
        let image = self.images.get(image_id.index())?;
        let event_index = self
            .detected_objects_by_image
            .get(image_id.index())
            .copied()
            .flatten()?;
        let event = self.events.get(event_index)?;
        match &event.kind {
            EventKind::DetectedObjects(frame) => Some((
                frame,
                frame
                    .image_time
                    .map(Time::to_wallclock)
                    .unwrap_or_else(|| image.display_time()),
            )),
            _ => None,
        }
    }
}

fn index_detected_objects_image_times(
    images: &[StereoImageIndex],
    images_by_embedded_time: &[usize],
    announcements: &BTreeMap<i64, (Time, SystemTime)>,
) -> BTreeMap<i64, DetectedObjectsImageTime> {
    announcements
        .iter()
        .filter_map(|(&sequence, &(source_time, _announcement_log_time))| {
            let image_index =
                nearest_image_index_by_embedded_time(images, images_by_embedded_time, source_time)?;
            let image = images.get(image_index)?;
            Some((
                sequence,
                DetectedObjectsImageTime {
                    publish_time: image.publish_time,
                },
            ))
        })
        .collect()
}

fn index_detected_objects_by_image(
    events: &[RecordedEvent],
    images: &[StereoImageIndex],
    images_by_embedded_time: &[usize],
    announcements: &BTreeMap<i64, (Time, SystemTime)>,
) -> Vec<Option<usize>> {
    let mut by_image = vec![None; images.len()];
    let mut fallback_image_index = 0;
    let use_stream_order_fallback = announcements.is_empty();

    for (event_index, event) in events.iter().enumerate() {
        let EventKind::DetectedObjects(frame) = &event.kind else {
            continue;
        };

        let image_index = announcements
            .get(&frame.sequence_number)
            .and_then(|time| {
                nearest_image_index_by_embedded_time(images, images_by_embedded_time, time.0)
            })
            .or_else(|| {
                frame.image_time.and_then(|time| {
                    first_image_index_at_or_after_embedded_time(
                        images,
                        images_by_embedded_time,
                        time,
                    )
                })
            })
            .or_else(|| {
                (use_stream_order_fallback && frame.image_time.is_none())
                    .then(|| {
                        let image_index = fallback_image_index;
                        fallback_image_index += 1;
                        (image_index < images.len()).then_some(image_index)
                    })
                    .flatten()
            });

        if let Some(image_index) = image_index
            && let Some(slot) = by_image.get_mut(image_index)
        {
            *slot = Some(event_index);
        }
    }

    by_image
}

fn index_detected_objects_sources_from_field_mark_associations(
    events: &[RecordedEvent],
    field_mark_associations: &[FieldMarkAssociationTiming],
) -> BTreeMap<i64, (Time, SystemTime)> {
    let mut pending_detected_objects = Vec::new();
    let mut sources = BTreeMap::new();
    let mut event_index = 0;

    for association in field_mark_associations {
        while let Some(event) = events.get(event_index) {
            if (nanos_since_epoch(event.log_time), event.order)
                > (nanos_since_epoch(association.log_time), association.order)
            {
                break;
            }

            if let EventKind::DetectedObjects(frame) = &event.kind {
                pending_detected_objects.push((frame.sequence_number, event.log_time));
            }
            event_index += 1;
        }

        let Some((pending_index, _)) = pending_detected_objects
            .iter()
            .enumerate()
            .min_by_key(|(_, (_, log_time))| nanos_abs_diff(*log_time, association.log_time))
        else {
            continue;
        };
        let (sequence, _) = pending_detected_objects.remove(pending_index);
        sources.insert(sequence, (association.source_time, association.log_time));
    }

    sources
}

fn nearest_image_index_by_embedded_time(
    images: &[StereoImageIndex],
    images_by_embedded_time: &[usize],
    time: Time,
) -> Option<usize> {
    let target_nanos = time.as_nanos();
    let next = images_by_embedded_time
        .partition_point(|&index| images[index].embedded_time.as_nanos() <= target_nanos);
    nearest_by_distance(
        next.checked_sub(1)
            .and_then(|index| images_by_embedded_time.get(index))
            .copied()
            .map(|index| {
                (
                    index,
                    u128::from(
                        images[index]
                            .embedded_time
                            .as_nanos()
                            .abs_diff(target_nanos),
                    ),
                )
            }),
        images_by_embedded_time.get(next).copied().map(|index| {
            (
                index,
                u128::from(
                    images[index]
                        .embedded_time
                        .as_nanos()
                        .abs_diff(target_nanos),
                ),
            )
        }),
    )
    .filter(|&index| {
        Duration::from_nanos(
            u128::from(
                images[index]
                    .embedded_time
                    .as_nanos()
                    .abs_diff(target_nanos),
            )
            .min(u64::MAX as u128) as u64,
        ) <= SNAPSHOT_MAX_TIME_DISTANCE
    })
}

fn first_image_index_at_or_after_embedded_time(
    images: &[StereoImageIndex],
    images_by_embedded_time: &[usize],
    time: Time,
) -> Option<usize> {
    let target_nanos = time.as_nanos();
    let next = images_by_embedded_time
        .partition_point(|&index| images[index].embedded_time.as_nanos() < target_nanos);
    images_by_embedded_time.get(next).copied().filter(|&index| {
        Duration::from_nanos(
            u128::from(
                images[index]
                    .embedded_time
                    .as_nanos()
                    .abs_diff(target_nanos),
            )
            .min(u64::MAX as u128) as u64,
        ) <= SNAPSHOT_MAX_TIME_DISTANCE
    })
}

#[derive(Clone)]
pub struct RecordedEvent {
    pub order: usize,
    pub log_time: SystemTime,
    pub publish_time: SystemTime,
    pub kind: EventKind,
}

impl RecordedEvent {
    pub fn display_time(&self) -> SystemTime {
        match &self.kind {
            EventKind::CameraMatrix(camera_matrix) => camera_matrix.time.to_wallclock(),
            EventKind::DetectedObjects(detected_objects) => detected_objects.display_time(),
            EventKind::RobotKinematics(robot_kinematics) => robot_kinematics.time.to_wallclock(),
            EventKind::FieldMarkAssociations(associations) => associations.time.to_wallclock(),
            EventKind::SolveDiagnostics(diagnostics) => diagnostics.time.to_wallclock(),
            EventKind::Imu(_)
            | EventKind::VisualOdometry(_)
            | EventKind::RecordedLocalization(_)
            | EventKind::CalibratedIntrinsics(_)
            | EventKind::GlobalLocalizationDebug(_) => self.publish_time,
        }
    }
}

struct FieldMarkAssociationTiming {
    order: usize,
    log_time: SystemTime,
    source_time: Time,
}

#[derive(Default)]
struct SnapshotIndex {
    camera_matrices: Vec<usize>,
    detected_objects: Vec<usize>,
    recorded_localizations: Vec<usize>,
    robot_kinematics: Vec<usize>,
    calibrated_intrinsics: Vec<usize>,
    field_mark_associations: Vec<usize>,
    global_localization_debug: Vec<usize>,
    solve_diagnostics: Vec<usize>,
}

impl SnapshotIndex {
    fn new(events: &[RecordedEvent]) -> Self {
        let mut index = Self::default();
        for (event_index, event) in events.iter().enumerate() {
            match &event.kind {
                EventKind::CameraMatrix(_) => index.camera_matrices.push(event_index),
                EventKind::DetectedObjects(_) => index.detected_objects.push(event_index),
                EventKind::RecordedLocalization(_) => {
                    index.recorded_localizations.push(event_index)
                }
                EventKind::RobotKinematics(_) => index.robot_kinematics.push(event_index),
                EventKind::CalibratedIntrinsics(_) => index.calibrated_intrinsics.push(event_index),
                EventKind::FieldMarkAssociations(_) => {
                    index.field_mark_associations.push(event_index)
                }
                EventKind::GlobalLocalizationDebug(_) => {
                    index.global_localization_debug.push(event_index)
                }
                EventKind::SolveDiagnostics(_) => index.solve_diagnostics.push(event_index),
                EventKind::Imu(_) | EventKind::VisualOdometry(_) => {}
            }
        }
        index.sort_by_display_time(events);
        index
    }

    fn sort_by_display_time(&mut self, events: &[RecordedEvent]) {
        for indexes in [
            &mut self.camera_matrices,
            &mut self.detected_objects,
            &mut self.recorded_localizations,
            &mut self.calibrated_intrinsics,
            &mut self.field_mark_associations,
            &mut self.global_localization_debug,
            &mut self.solve_diagnostics,
        ] {
            indexes.sort_by_key(|&index| nanos_since_epoch(events[index].display_time()));
        }
    }

    fn nearest_camera_matrix<'a>(
        &self,
        events: &'a [RecordedEvent],
        time: SystemTime,
    ) -> Option<&'a RecordedEvent> {
        Self::nearest_with_max(
            events,
            &self.camera_matrices,
            time,
            SNAPSHOT_MAX_TIME_DISTANCE,
        )
    }

    fn nearest_detected_objects<'a>(
        &self,
        events: &'a [RecordedEvent],
        time: SystemTime,
    ) -> Option<&'a RecordedEvent> {
        Self::nearest_with_max(
            events,
            &self.detected_objects,
            time,
            SNAPSHOT_MAX_TIME_DISTANCE,
        )
    }

    fn latest_recorded_localization<'a>(
        &self,
        events: &'a [RecordedEvent],
        time: SystemTime,
    ) -> Option<&'a RecordedEvent> {
        Self::latest_with_max(
            events,
            &self.recorded_localizations,
            time,
            SNAPSHOT_MAX_TIME_DISTANCE,
        )
    }

    fn nearest_robot_kinematics<'a>(
        &self,
        events: &'a [RecordedEvent],
        time: SystemTime,
    ) -> Option<&'a RecordedEvent> {
        Self::nearest_with_max(
            events,
            &self.robot_kinematics,
            time,
            SNAPSHOT_MAX_TIME_DISTANCE,
        )
    }

    fn latest_calibrated_intrinsics<'a>(
        &self,
        events: &'a [RecordedEvent],
        time: SystemTime,
    ) -> Option<&'a RecordedEvent> {
        Self::latest_with_max(
            events,
            &self.calibrated_intrinsics,
            time,
            SNAPSHOT_MAX_TIME_DISTANCE,
        )
    }

    fn nearest_field_mark_associations<'a>(
        &self,
        events: &'a [RecordedEvent],
        time: SystemTime,
    ) -> Option<&'a RecordedEvent> {
        Self::nearest_with_max(
            events,
            &self.field_mark_associations,
            time,
            SNAPSHOT_MAX_TIME_DISTANCE,
        )
    }

    fn nearest_global_localization_debug<'a>(
        &self,
        events: &'a [RecordedEvent],
        time: SystemTime,
    ) -> Option<&'a RecordedEvent> {
        Self::nearest_with_max(
            events,
            &self.global_localization_debug,
            time,
            SNAPSHOT_MAX_TIME_DISTANCE,
        )
    }

    fn latest_solve_diagnostics<'a>(
        &self,
        events: &'a [RecordedEvent],
        time: SystemTime,
    ) -> Option<&'a RecordedEvent> {
        Self::latest_with_max(
            events,
            &self.solve_diagnostics,
            time,
            SNAPSHOT_MAX_TIME_DISTANCE,
        )
    }

    fn latest_with_max<'a>(
        events: &'a [RecordedEvent],
        indexes: &[usize],
        time: SystemTime,
        max_age: Duration,
    ) -> Option<&'a RecordedEvent> {
        let next = indexes.partition_point(|&index| events[index].display_time() <= time);
        next.checked_sub(1)
            .and_then(|index| indexes.get(index))
            .and_then(|&index| events.get(index))
            .filter(|event| abs_duration(event.display_time(), time) <= max_age)
    }

    fn nearest_with_max<'a>(
        events: &'a [RecordedEvent],
        indexes: &[usize],
        time: SystemTime,
        max_distance: Duration,
    ) -> Option<&'a RecordedEvent> {
        let next = indexes.partition_point(|&index| events[index].display_time() <= time);
        nearest_by_distance(
            next.checked_sub(1)
                .and_then(|index| indexes.get(index))
                .map(|&index| {
                    (
                        &events[index],
                        nanos_abs_diff(events[index].display_time(), time),
                    )
                }),
            indexes.get(next).map(|&index| {
                (
                    &events[index],
                    nanos_abs_diff(events[index].display_time(), time),
                )
            }),
        )
        .filter(|event| abs_duration(event.display_time(), time) <= max_distance)
    }
}

#[derive(Clone)]
pub enum EventKind {
    Imu(ImuState),
    VisualOdometry(VisualOdometryDelta),
    RobotKinematics(Box<TimeWrapper<RobotKinematics>>),
    CameraMatrix(TimeWrapper<CameraMatrix>),
    DetectedObjects(DetectedObjectsFrame),
    RecordedLocalization(Option<Isometry3<Field, Robot>>),
    CalibratedIntrinsics(Intrinsic),
    FieldMarkAssociations(TimeWrapper<FieldMarkAssociations>),
    GlobalLocalizationDebug(Option<GlobalLocalizationDebug>),
    SolveDiagnostics(TimeWrapper<SolveDiagnostics>),
}

#[derive(Clone)]
pub struct DetectedObjectsFrame {
    pub objects: Vec<Object<RobocupObjectLabel>>,
    pub image_time: Option<Time>,
    pub publish_time: SystemTime,
    pub announcement_log_time: Option<SystemTime>,
    pub sequence_number: i64,
}

impl DetectedObjectsFrame {
    pub fn display_time(&self) -> SystemTime {
        self.image_time
            .map(Time::to_wallclock)
            .unwrap_or(self.publish_time)
    }
}

pub struct StereoImageIndex {
    pub order: usize,
    pub log_time: SystemTime,
    pub publish_time: SystemTime,
    pub embedded_time: Time,
    data: Arc<[u8]>,
}

impl StereoImageIndex {
    fn display_time(&self) -> SystemTime {
        self.embedded_time.to_wallclock()
    }
}

struct DetectedObjectsImageTime {
    publish_time: SystemTime,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StereoImageId(usize);

impl StereoImageId {
    pub fn index(self) -> usize {
        self.0
    }
}

impl fmt::Display for StereoImageId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.0)
    }
}

#[derive(Clone, Default)]
pub struct RecordingSnapshot {
    pub image_id: Option<StereoImageId>,
    pub image_display_time: Option<SystemTime>,
    pub camera_matrix: Option<TimeWrapper<CameraMatrix>>,
    pub detected_objects: Vec<Object<RobocupObjectLabel>>,
    pub detected_objects_time: Option<SystemTime>,
    pub detected_objects_frame: Option<DetectedObjectsFrame>,
    pub recorded_localization: Option<Isometry3<Field, Robot>>,
    pub robot_kinematics: Option<TimeWrapper<Arc<RobotKinematics>>>,
    pub calibrated_intrinsics: Option<Intrinsic>,
    pub calibrated_intrinsics_time: Option<SystemTime>,
    pub field_mark_associations: Option<TimeWrapper<FieldMarkAssociations>>,
    pub global_localization_debug: Option<GlobalLocalizationDebug>,
    pub global_localization_debug_time: Option<SystemTime>,
    pub solve_diagnostics: Option<TimeWrapper<SolveDiagnostics>>,
}

#[derive(Clone)]
pub struct StereoFrame {
    pub sequence: u64,
    pub source_time: Time,
    pub log_time: SystemTime,
    pub publish_time: SystemTime,
    pub left: CameraImage,
    pub right: CameraImage,
}

#[derive(Clone)]
pub struct CameraImage {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct TrajectoryPoint {
    pub seconds: f64,
    pub robot_to_field: Isometry3<Robot, Field, f64>,
    pub segment_id: u64,
}

fn camera_image_from_ros(image: RosImage) -> Result<CameraImage> {
    let rgb: RgbImage = image
        .try_into()
        .map_err(|error| color_eyre::eyre::eyre!("failed to decode ROS image: {error}"))?;
    let (width, height) = rgb.dimensions();
    let mut rgba = Vec::with_capacity(width as usize * height as usize * 4);
    for pixel in rgb.pixels() {
        rgba.extend_from_slice(&[pixel[0], pixel[1], pixel[2], 255]);
    }
    Ok(CameraImage {
        width,
        height,
        rgba,
    })
}

fn decode_message<T>(data: &[u8]) -> Result<T>
where
    T: DeserializeOwned,
{
    let (value, _consumed) = from_bytes::<T, LittleEndian>(cdr_payload(data))?;
    Ok(value)
}

fn cdr_payload(data: &[u8]) -> &[u8] {
    if data.len() >= 4 && matches!(&data[..4], [0, 1, 0, 0] | [0, 0, 0, 0]) {
        &data[4..]
    } else {
        data
    }
}

fn decode_time_prefix(data: &[u8]) -> Result<Time> {
    let data = cdr_payload(data);
    if data.len() < 12 {
        return Err(color_eyre::eyre::eyre!("payload too short for Time prefix"));
    }
    let secs = u64::from_le_bytes(data[0..8].try_into()?);
    let nanos = u32::from_le_bytes(data[8..12].try_into()?);
    let total_nanos = secs
        .saturating_mul(1_000_000_000)
        .saturating_add(u64::from(nanos));
    Ok(Time::from_nanos(total_nanos.min(i64::MAX as u64) as i64))
}

#[derive(Deserialize)]
struct WireTimeWrapper<T> {
    time: Time,
    inner: T,
}

#[derive(Deserialize)]
struct WireCameraMatrix {
    ground_to_robot: WireIsometry3,
    robot_to_head: WireIsometry3,
    head_to_camera: WireIsometry3,
    intrinsics: WireIntrinsic,
    field_of_view: [f32; 2],
    horizon: Option<WireHorizon>,
    image_size: [f32; 2],
}

#[derive(Deserialize)]
struct WireIntrinsic {
    focals: [f32; 2],
    optical_center: [f32; 2],
}

#[derive(Deserialize)]
struct WireHorizon {
    vanishing_point: [f32; 2],
    normal: [f32; 2],
}

#[derive(Deserialize)]
struct WireVisualOdometryDelta {
    previous_time: Time,
    current_time: Time,
    current_left_camera_to_previous_left_camera: WireIsometry3,
}

#[derive(Deserialize)]
struct WireAnnouncement {
    time: Time,
    #[allow(dead_code)]
    source_global_id: ros_z::EndpointGlobalId,
    sequence_number: i64,
}

#[derive(Deserialize)]
struct WireIsometry3 {
    rotation: [f32; 4],
    translation: [f32; 3],
}

impl WireCameraMatrix {
    fn into_camera_matrix(self) -> CameraMatrix {
        let image_size: linear_algebra::Vector2<Pixel> =
            vector![self.image_size[0], self.image_size[1]];
        let normalized_focal = nalgebra::vector![
            self.intrinsics.focals[0] / image_size.inner.x,
            self.intrinsics.focals[1] / image_size.inner.y,
        ];
        let normalized_center = nalgebra::point![
            self.intrinsics.optical_center[0] / image_size.inner.x,
            self.intrinsics.optical_center[1] / image_size.inner.y,
        ];
        let _ = self.field_of_view;
        if let Some(horizon) = self.horizon {
            let _ = (horizon.vanishing_point, horizon.normal);
        }

        CameraMatrix::from_normalized_focal_and_center(
            normalized_focal,
            normalized_center,
            image_size,
            self.ground_to_robot.framed(),
            self.robot_to_head.framed(),
            self.head_to_camera.framed(),
        )
    }
}

impl WireVisualOdometryDelta {
    fn into_visual_odometry_delta(self) -> VisualOdometryDelta {
        VisualOdometryDelta {
            previous_time: self.previous_time,
            current_time: self.current_time,
            current_left_camera_to_previous_left_camera: self
                .current_left_camera_to_previous_left_camera
                .into_isometry(),
        }
    }
}

impl WireIsometry3 {
    fn into_isometry(self) -> nalgebra::Isometry3<f32> {
        let rotation = nalgebra::UnitQuaternion::new_normalize(nalgebra::Quaternion::new(
            self.rotation[3],
            self.rotation[0],
            self.rotation[1],
            self.rotation[2],
        ));
        nalgebra::Isometry3::from_parts(
            nalgebra::Translation3::new(
                self.translation[0],
                self.translation[1],
                self.translation[2],
            ),
            rotation,
        )
    }

    fn framed<From, To>(self) -> linear_algebra::Isometry3<From, To> {
        self.into_isometry().framed_transform()
    }
}

fn decode_recorded_camera_matrix(message: &Message<'_>) -> Result<TimeWrapper<CameraMatrix>> {
    let wire: WireTimeWrapper<WireCameraMatrix> = decode_recorded_message(message)?;
    Ok(TimeWrapper {
        time: wire.time,
        inner: wire.inner.into_camera_matrix(),
    })
}

fn decode_recorded_visual_odometry(message: &Message<'_>) -> Result<VisualOdometryDelta> {
    let wire: WireVisualOdometryDelta = decode_recorded_message(message)?;
    Ok(wire.into_visual_odometry_delta())
}

fn decode_recorded_message<T>(message: &Message<'_>) -> Result<T>
where
    T: DeserializeOwned,
{
    decode_message(&message.data).wrap_err_with(|| {
        format!(
            "failed to decode topic {} sequence {}",
            message.channel.topic, message.sequence
        )
    })
}

fn system_time_from_nanos(nanos: u64) -> SystemTime {
    UNIX_EPOCH + Duration::from_nanos(nanos)
}

pub fn nanos_since_epoch(time: SystemTime) -> u128 {
    match time.duration_since(UNIX_EPOCH) {
        Ok(duration) => duration.as_nanos(),
        Err(_) => 0,
    }
}

pub fn nanos_abs_diff(a: SystemTime, b: SystemTime) -> u128 {
    nanos_since_epoch(a).abs_diff(nanos_since_epoch(b))
}

pub fn abs_duration(a: SystemTime, b: SystemTime) -> Duration {
    a.duration_since(b)
        .or_else(|_| b.duration_since(a))
        .unwrap_or(Duration::ZERO)
}

pub fn seconds_since(time: SystemTime, start: SystemTime) -> f64 {
    match time.duration_since(start) {
        Ok(duration) => duration.as_secs_f64(),
        Err(error) => -error.duration().as_secs_f64(),
    }
}

#[cfg(test)]
mod tests {
    use std::{path::Path, sync::Arc};

    use super::*;

    #[test]
    fn detected_objects_display_time_prefers_announced_image_time() {
        let publish_time = UNIX_EPOCH + Duration::from_secs(10);
        let image_time = Time::from_wallclock(UNIX_EPOCH + Duration::from_secs(12));
        let frame = DetectedObjectsFrame {
            objects: Vec::new(),
            image_time: Some(image_time),
            publish_time,
            announcement_log_time: None,
            sequence_number: 7,
        };

        assert_eq!(frame.display_time(), image_time.to_wallclock());
    }

    #[test]
    fn snapshot_index_rejects_stale_detected_objects() {
        let image_time = UNIX_EPOCH + Duration::from_secs(10);
        let frame = DetectedObjectsFrame {
            objects: Vec::new(),
            image_time: Some(Time::from_wallclock(image_time)),
            publish_time: image_time,
            announcement_log_time: None,
            sequence_number: 1,
        };
        let events = vec![RecordedEvent {
            order: 0,
            log_time: image_time,
            publish_time: image_time,
            kind: EventKind::DetectedObjects(frame),
        }];
        let index = SnapshotIndex::new(&events);

        assert!(
            index
                .nearest_detected_objects(&events, image_time + SNAPSHOT_MAX_TIME_DISTANCE / 2)
                .is_some()
        );
        assert!(
            index
                .nearest_detected_objects(
                    &events,
                    image_time + SNAPSHOT_MAX_TIME_DISTANCE + Duration::from_millis(1),
                )
                .is_none()
        );
    }

    #[test]
    #[ignore = "loads the large repository localization recording"]
    fn loads_repository_recording() -> Result<()> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../recording.mcap");
        let recording = Recording::load(&path)?;

        assert!(recording.event_count() > 0);
        assert!(recording.image_count() > 0);
        assert!(recording.topic_counts.contains_key(TOPIC_CAMERA_MATRIX));
        assert!(recording.topic_counts.contains_key(TOPIC_STEREO_IMAGE_PAIR));

        let frame = recording.decode_stereo_image(StereoImageId(0))?;
        assert!(frame.left.width > 0);
        assert!(frame.left.height > 0);
        assert_eq!(
            frame.left.rgba.len(),
            frame.left.width as usize * frame.left.height as usize * 4,
        );

        Ok(())
    }

    #[test]
    fn detection_index_uses_announced_image_time() {
        let images = vec![test_image(0, 10), test_image(1, 20), test_image(2, 30)];
        let images_by_embedded_time = vec![0, 1, 2];
        let events = vec![test_detected_objects_event(7, Some(Time::from_nanos(20)))];

        let index = index_detected_objects_by_image(
            &events,
            &images,
            &images_by_embedded_time,
            &BTreeMap::new(),
        );

        assert_eq!(index, vec![None, Some(0), None]);
    }

    #[test]
    fn detection_index_uses_following_image_when_previous_is_closer() {
        let images = vec![test_image(0, 90_000_000), test_image(1, 120_000_000)];
        let images_by_embedded_time = vec![0, 1];
        let events = vec![test_detected_objects_event(
            7,
            Some(Time::from_nanos(100_000_000)),
        )];

        let index = index_detected_objects_by_image(
            &events,
            &images,
            &images_by_embedded_time,
            &BTreeMap::new(),
        );

        assert_eq!(index, vec![None, Some(0)]);
    }

    #[test]
    fn detection_index_rejects_stale_following_image() {
        let images = vec![test_image(0, 90_000_000), test_image(1, 250_000_001)];
        let images_by_embedded_time = vec![0, 1];
        let events = vec![test_detected_objects_event(
            7,
            Some(Time::from_nanos(100_000_000)),
        )];

        let index = index_detected_objects_by_image(
            &events,
            &images,
            &images_by_embedded_time,
            &BTreeMap::new(),
        );

        assert_eq!(index, vec![None, None]);
    }

    #[test]
    fn detection_index_uses_stream_order_without_timing_sources() {
        let images = vec![test_image(0, 10), test_image(1, 20)];
        let images_by_embedded_time = vec![0, 1];
        let events = vec![
            test_detected_objects_event(0, None),
            test_detected_objects_event(1, None),
        ];

        let index = index_detected_objects_by_image(
            &events,
            &images,
            &images_by_embedded_time,
            &BTreeMap::new(),
        );

        assert_eq!(index, vec![Some(0), Some(1)]);
    }

    #[test]
    fn detection_index_skips_unmapped_detections_when_timing_sources_exist() {
        let images = vec![test_image(0, 10), test_image(1, 20)];
        let images_by_embedded_time = vec![0, 1];
        let events = vec![
            test_detected_objects_event(1, None),
            test_detected_objects_event(2, None),
        ];
        let announcements = BTreeMap::from([(2, (Time::from_nanos(20), UNIX_EPOCH))]);

        let index = index_detected_objects_by_image(
            &events,
            &images,
            &images_by_embedded_time,
            &announcements,
        );

        assert_eq!(index, vec![None, Some(1)]);
    }

    #[test]
    fn detection_index_uses_field_mark_association_time_without_announcements() {
        let images = vec![test_image(0, 10), test_image(1, 20), test_image(2, 30)];
        let images_by_embedded_time = vec![0, 1, 2];
        let events = vec![test_detected_objects_event_at(42, 19)];
        let field_mark_associations = vec![test_field_mark_association_at(20, 21)];
        let detected_objects_image_sources =
            index_detected_objects_sources_from_field_mark_associations(
                &events,
                &field_mark_associations,
            );

        let index = index_detected_objects_by_image(
            &events,
            &images,
            &images_by_embedded_time,
            &detected_objects_image_sources,
        );

        assert_eq!(
            detected_objects_image_sources.get(&42),
            Some(&(Time::from_nanos(20), system_time_from_nanos(21)))
        );
        assert_eq!(index, vec![None, Some(0), None]);
    }

    #[test]
    fn field_mark_association_time_matches_nearest_pending_detection() {
        let events = vec![
            test_detected_objects_event_at(10, 10),
            test_detected_objects_event_at(20, 20),
        ];
        let field_mark_associations = vec![test_field_mark_association_at(20, 21)];

        let sources = index_detected_objects_sources_from_field_mark_associations(
            &events,
            &field_mark_associations,
        );

        assert_eq!(
            sources.get(&20),
            Some(&(Time::from_nanos(20), system_time_from_nanos(21)))
        );
        assert!(!sources.contains_key(&10));
    }

    fn test_image(order: usize, embedded_nanos: i64) -> StereoImageIndex {
        let time = system_time_from_nanos(embedded_nanos as u64);
        StereoImageIndex {
            order,
            log_time: time,
            publish_time: time,
            embedded_time: Time::from_nanos(embedded_nanos),
            data: Arc::from([]),
        }
    }

    fn test_detected_objects_event(sequence: i64, image_time: Option<Time>) -> RecordedEvent {
        RecordedEvent {
            order: sequence as usize,
            log_time: system_time_from_nanos(sequence as u64),
            publish_time: system_time_from_nanos(sequence as u64),
            kind: EventKind::DetectedObjects(DetectedObjectsFrame {
                objects: Vec::new(),
                image_time,
                publish_time: system_time_from_nanos(sequence as u64),
                announcement_log_time: None,
                sequence_number: sequence,
            }),
        }
    }

    fn test_detected_objects_event_at(sequence: i64, log_nanos: i64) -> RecordedEvent {
        RecordedEvent {
            order: sequence as usize,
            log_time: system_time_from_nanos(log_nanos as u64),
            publish_time: system_time_from_nanos(log_nanos as u64),
            kind: EventKind::DetectedObjects(DetectedObjectsFrame {
                objects: Vec::new(),
                image_time: None,
                publish_time: system_time_from_nanos(log_nanos as u64),
                announcement_log_time: None,
                sequence_number: sequence,
            }),
        }
    }

    fn test_field_mark_association_at(
        embedded_nanos: i64,
        log_nanos: i64,
    ) -> FieldMarkAssociationTiming {
        FieldMarkAssociationTiming {
            order: embedded_nanos as usize,
            log_time: system_time_from_nanos(log_nanos as u64),
            source_time: Time::from_nanos(embedded_nanos),
        }
    }
}
