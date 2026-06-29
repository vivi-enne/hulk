use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::Sender,
    },
    time::{Duration, Instant, SystemTime},
};

use color_eyre::Result;
use coordinate_systems::{Field, Robot};
use field_mark_association::{
    FieldMarkAssociation, FieldMarkAssociationKind, FieldMarkAssociationParameters,
    FieldMarkAssociationState, FieldMarkAssociations, GlobalLocalizationDebugStatus,
    GlobalLocalizerParameters, PoseHintAssociationParameters, find_detected_visual_features,
};
use linear_algebra::IntoTransform;
use localization_3d::{
    Localization3dParameters, VisualAnchorPoseGate, VisualOdometryExtrinsicGate,
    backend_configuration_from_parameters, ingest_foot_heights, ingest_visual_odometry,
    should_reject_unanchored_pose_update, should_reject_visual_odometry_due_to_head_motion,
};
use localization_factrs::{
    BackendConfiguration, VinsBackend, VinsFrontend, VisualReprojectionAssociation,
    VisualReprojectionAssociationKind, backend::BackendSolveDiagnostics, initialize,
};
use nalgebra::SMatrix;
use projection::camera_matrix::CameraMatrix;
use ros_z::time::Time;
use types::{
    field_dimensions::FieldDimensions, time_wrapper::TimeWrapper,
    visual_odometry::VisualOdometryDelta,
};

use crate::mcap_recording::{
    EventKind, RecordedEvent, Recording, TRAJECTORY_MAX_SAMPLE_GAP_SECONDS, TrajectoryPoint,
    nanos_abs_diff, nanos_since_epoch, seconds_since,
};

const CAMERA_MATRIX_MAX_TIME_DISTANCE: Duration = Duration::from_millis(100);
const DEFAULT_POSE_HINT_VISUAL_MIN_FEATURES_PER_FRAME: usize = 3;

#[derive(Clone, Debug, PartialEq)]
pub struct ReplayParameters {
    pub timestamp_mode: TimestampMode,
    pub solve_cadence_ms: f64,
    pub optimizer_iterations: usize,
    pub max_window_seconds: f64,
    pub solve_start_seconds: f64,
    pub solve_end_seconds: f64,
    pub visual_feature_noise_variance: f64,
    pub pose_hint_visual_feature_noise_variance: f64,
    pub pose_hint_visual_huber_threshold: f64,
    pub pose_hint_visual_min_features_per_frame: usize,
    pub visual_odometry_covariance: f64,
    pub override_visual_odometry_covariance: bool,
    pub max_visual_odometry_translation: Option<f32>,
    pub max_visual_odometry_rotation: Option<f32>,
    pub include_visual_odometry: bool,
    pub reject_visual_odometry_during_head_motion: bool,
    pub max_visual_odometry_extrinsic_rotation: f32,
    pub max_visual_odometry_extrinsic_translation: f32,
    pub require_recent_visual_anchor_for_large_pose_updates: bool,
    pub max_visual_anchor_age: Duration,
    pub max_unanchored_translation_update: f32,
    pub max_unanchored_yaw_update: f32,
    pub include_global_features: bool,
    pub include_imu: bool,
    pub include_imu_kinematics: bool,
    pub include_imu_roll_pitch: bool,
    pub include_imu_yaw: bool,
    pub include_current_spline_orientation: bool,
    pub include_foot_heights: bool,
    pub recompute_global_features: bool,
    pub global_localizer: GlobalLocalizerParameters,
    pub pose_hint: PoseHintAssociationParameters,
}

#[derive(Clone, Debug)]
pub struct ReplayVisualOdometryEvent {
    pub log_time: SystemTime,
    pub publish_time: SystemTime,
    pub delta: VisualOdometryDelta,
}

enum ReplayEvent<'a> {
    Recorded(&'a RecordedEvent),
    VisualOdometryOverride(&'a ReplayVisualOdometryEvent),
}

impl ReplayEvent<'_> {
    fn log_time(&self) -> SystemTime {
        match self {
            Self::Recorded(event) => event.log_time,
            Self::VisualOdometryOverride(event) => event.log_time,
        }
    }
}

impl Default for ReplayParameters {
    fn default() -> Self {
        let localization_parameters = Localization3dParameters::default();
        let association_parameters = FieldMarkAssociationParameters::default();
        Self {
            timestamp_mode: TimestampMode::Embedded,
            solve_cadence_ms: 30.0,
            optimizer_iterations: 5,
            max_window_seconds: 3.0,
            solve_start_seconds: 0.0,
            solve_end_seconds: f64::INFINITY,
            visual_feature_noise_variance: localization_parameters.visual_feature_noise_variance,
            pose_hint_visual_feature_noise_variance: localization_parameters
                .pose_hint_visual_feature_noise_variance,
            pose_hint_visual_huber_threshold: localization_parameters
                .pose_hint_visual_huber_threshold,
            pose_hint_visual_min_features_per_frame:
                DEFAULT_POSE_HINT_VISUAL_MIN_FEATURES_PER_FRAME,
            visual_odometry_covariance: 1.0e-2,
            override_visual_odometry_covariance: false,
            max_visual_odometry_translation: None,
            max_visual_odometry_rotation: None,
            include_visual_odometry: true,
            reject_visual_odometry_during_head_motion: localization_parameters
                .reject_visual_odometry_during_head_motion,
            max_visual_odometry_extrinsic_rotation: localization_parameters
                .max_visual_odometry_extrinsic_rotation,
            max_visual_odometry_extrinsic_translation: localization_parameters
                .max_visual_odometry_extrinsic_translation,
            require_recent_visual_anchor_for_large_pose_updates: localization_parameters
                .require_recent_visual_anchor_for_large_pose_updates,
            max_visual_anchor_age: localization_parameters.max_visual_anchor_age,
            max_unanchored_translation_update: localization_parameters
                .max_unanchored_translation_update,
            max_unanchored_yaw_update: localization_parameters.max_unanchored_yaw_update,
            include_global_features: true,
            include_imu: true,
            include_imu_kinematics: true,
            include_imu_roll_pitch: true,
            include_imu_yaw: true,
            include_current_spline_orientation: true,
            include_foot_heights: true,
            recompute_global_features: false,
            global_localizer: association_parameters.global_localizer,
            pose_hint: association_parameters.pose_hint,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TimestampMode {
    McapPublish,
    Embedded,
}

#[derive(Clone, Debug)]
pub enum ResolveMessage {
    Progress(ResolveProgress),
    Finished(ResolveResult),
    Failed(String),
    Cancelled,
}

#[derive(Clone, Debug)]
pub struct ResolveProgress {
    pub processed_events: usize,
    pub total_events: usize,
    pub solve_count: usize,
}

#[derive(Clone, Debug)]
pub struct ResolveResult {
    pub parameters: ReplayParameters,
    pub samples: Vec<SolveSample>,
    pub vo_trajectory: Vec<TrajectoryPoint>,
    pub stats: ReplayStats,
    pub elapsed: Duration,
}

impl ResolveResult {
    pub fn trajectory(&self) -> Vec<TrajectoryPoint> {
        let mut segment_id = 0;
        let mut previous_seconds = None;
        self.samples
            .iter()
            .map(|sample| {
                if let Some(previous_seconds) = previous_seconds
                    && sample.graph_seconds - previous_seconds > TRAJECTORY_MAX_SAMPLE_GAP_SECONDS
                {
                    segment_id += 1;
                }
                previous_seconds = Some(sample.graph_seconds);
                TrajectoryPoint {
                    seconds: sample.graph_seconds,
                    robot_to_field: sample.robot_to_field,
                    segment_id,
                }
            })
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct SolveSample {
    pub replay_seconds: f64,
    pub graph_seconds: f64,
    pub solve_duration: Duration,
    pub robot_to_field: linear_algebra::Isometry3<Robot, Field, f64>,
    pub diagnostics: Option<BackendSolveDiagnostics>,
    pub stats: ReplayStats,
}

#[derive(Clone, Debug, Default)]
pub struct ReplayStats {
    pub imu_ingested: usize,
    pub foot_heights_ingested: usize,
    pub vo_received: usize,
    pub vo_ingested: usize,
    pub vo_dropped_invalid: usize,
    pub vo_dropped_gated: usize,
    pub vo_skipped_missing_camera_matrix: usize,
    pub vo_skipped_stale_camera_matrix: usize,
    pub vo_skipped_head_motion: usize,
    pub pose_updates_reused_unanchored: usize,
    pub global_frames: usize,
    pub global_candidates: usize,
    pub global_none: usize,
    pub global_ambiguous: usize,
    pub global_unique_modulo_symmetry: usize,
    pub global_frames_ingested: usize,
    pub global_associations_ingested: usize,
    pub global_unique_frames_ingested: usize,
    pub global_unique_associations_ingested: usize,
    pub pose_hint_frames_ingested: usize,
    pub pose_hint_associations_ingested: usize,
}

pub fn spawn_resolve(
    recording: Arc<Recording>,
    parameters: ReplayParameters,
    cancelled: Arc<AtomicBool>,
    sender: Sender<ResolveMessage>,
) {
    std::thread::spawn(move || {
        let result = run_resolve(&recording, parameters, None, &cancelled, &sender);
        let message = match result {
            Ok(Some(result)) => ResolveMessage::Finished(result),
            Ok(None) => ResolveMessage::Cancelled,
            Err(error) => ResolveMessage::Failed(format!("{error:#}")),
        };
        let _ = sender.send(message);
    });
}

pub fn resolve_recording(
    recording: &Recording,
    parameters: ReplayParameters,
) -> Result<ResolveResult> {
    let cancelled = AtomicBool::new(false);
    let (sender, _receiver) = std::sync::mpsc::channel();
    run_resolve(recording, parameters, None, &cancelled, &sender)?
        .ok_or_else(|| color_eyre::eyre::eyre!("resolve was cancelled"))
}

pub fn resolve_recording_with_visual_odometry_override(
    recording: &Recording,
    parameters: ReplayParameters,
    visual_odometry_override: &[ReplayVisualOdometryEvent],
) -> Result<ResolveResult> {
    let cancelled = AtomicBool::new(false);
    let (sender, _receiver) = std::sync::mpsc::channel();
    run_resolve(
        recording,
        parameters,
        Some(visual_odometry_override),
        &cancelled,
        &sender,
    )?
    .ok_or_else(|| color_eyre::eyre::eyre!("resolve was cancelled"))
}

fn run_resolve(
    recording: &Recording,
    parameters: ReplayParameters,
    visual_odometry_override: Option<&[ReplayVisualOdometryEvent]>,
    cancelled: &AtomicBool,
    sender: &Sender<ResolveMessage>,
) -> Result<Option<ResolveResult>> {
    let started = Instant::now();
    let initial_state =
        localization_3d::initial_state_from_camera_matrix(&recording.first_camera_matrix);
    let (mut frontend, mut backend) = initialize(backend_config(&parameters), initial_state);
    let field_dimensions = recording
        .field_dimensions
        .unwrap_or(FieldDimensions::SPL_2025);
    let mut camera_matrices = OnlineCameraMatrices::default();
    let mut vo_timestamps = VisualOdometryTimestampTracker::default();
    let mut association_state = FieldMarkAssociationState::default();
    let mut stats = ReplayStats::default();
    let mut samples = Vec::new();
    let mut vo_only = VisualOdometryTrajectory::new(
        localization_3d::initial_robot_to_field_from_camera_matrix(&recording.first_camera_matrix)
            .inner,
    );
    let mut has_pending_measurements = false;
    let mut last_visual_anchor_time = None;
    let mut last_accepted_robot_to_field = None;
    let cadence = Duration::from_secs_f64((parameters.solve_cadence_ms / 1000.0).max(0.001));
    let range_start_seconds = parameters.solve_start_seconds.max(0.0);
    let range_end_seconds = parameters
        .solve_end_seconds
        .min(recording.duration().as_secs_f64())
        .max(range_start_seconds);
    let range_start_time =
        recording.start_log_time() + Duration::from_secs_f64(range_start_seconds);
    let range_end_time = recording.start_log_time() + Duration::from_secs_f64(range_end_seconds);
    let mut next_solve_time = range_start_time + cadence;
    let replay_events = merged_replay_events(recording, visual_odometry_override);
    let total_events = replay_events.len();
    let has_recorded_global_features = recording
        .events()
        .iter()
        .any(|event| matches!(event.kind, EventKind::FieldMarkAssociations(_)));
    let recompute_global_features =
        parameters.recompute_global_features || !has_recorded_global_features;

    for (index, replay_event) in replay_events.iter().enumerate() {
        if cancelled.load(Ordering::Relaxed) {
            return Ok(None);
        }
        let replay_time = replay_event.log_time();
        if replay_time > range_end_time {
            break;
        }
        if replay_time < range_start_time {
            match replay_event {
                ReplayEvent::Recorded(event) => match &event.kind {
                    EventKind::CameraMatrix(camera_matrix) => {
                        camera_matrices.push(event, camera_matrix.clone());
                    }
                    EventKind::VisualOdometry(delta) => {
                        let _ = vo_timestamps.measurement_times(
                            event,
                            delta,
                            parameters.timestamp_mode,
                            recording,
                        );
                    }
                    _ => {}
                },
                ReplayEvent::VisualOdometryOverride(visual_odometry) => {
                    let event = RecordedEvent {
                        order: 0,
                        log_time: visual_odometry.log_time,
                        publish_time: visual_odometry.publish_time,
                        kind: EventKind::VisualOdometry(visual_odometry.delta.clone()),
                    };
                    let _ = vo_timestamps.measurement_times(
                        &event,
                        &visual_odometry.delta,
                        parameters.timestamp_mode,
                        recording,
                    );
                }
            }
            continue;
        }

        while next_solve_time <= replay_time {
            if has_pending_measurements {
                solve_and_record(
                    &mut backend,
                    &mut frontend,
                    recording,
                    parameters.timestamp_mode,
                    next_solve_time,
                    &mut stats,
                    &parameters,
                    last_visual_anchor_time,
                    &mut last_accepted_robot_to_field,
                    &mut samples,
                )?;
                has_pending_measurements = false;
            }
            next_solve_time += cadence;
        }

        match replay_event {
            ReplayEvent::VisualOdometryOverride(visual_odometry)
                if parameters.include_visual_odometry =>
            {
                let event = RecordedEvent {
                    order: 0,
                    log_time: visual_odometry.log_time,
                    publish_time: visual_odometry.publish_time,
                    kind: EventKind::VisualOdometry(visual_odometry.delta.clone()),
                };
                ingest_vo_event(
                    recording,
                    &event,
                    &visual_odometry.delta,
                    &parameters,
                    &mut vo_timestamps,
                    &camera_matrices,
                    &mut frontend,
                    &mut stats,
                    &mut vo_only,
                    &mut has_pending_measurements,
                )?;
            }
            ReplayEvent::VisualOdometryOverride(_) => {
                stats.vo_received += 1;
            }
            ReplayEvent::Recorded(event) => match &event.kind {
                EventKind::Imu(imu) if parameters.include_imu => {
                    frontend.ingest_imu(event.publish_time, *imu)?;
                    stats.imu_ingested += 1;
                    has_pending_measurements = true;
                }
                EventKind::CameraMatrix(camera_matrix) => {
                    camera_matrices.push(event, camera_matrix.clone());
                }
                EventKind::RobotKinematics(robot_kinematics) if parameters.include_foot_heights => {
                    let mut robot_kinematics = robot_kinematics.as_ref().clone();
                    if parameters.timestamp_mode == TimestampMode::McapPublish {
                        robot_kinematics.time = Time::from_wallclock(event.publish_time);
                    }
                    ingest_foot_heights(&mut frontend, robot_kinematics)?;
                    stats.foot_heights_ingested += 1;
                    has_pending_measurements = true;
                }
                EventKind::VisualOdometry(delta) if parameters.include_visual_odometry => {
                    ingest_vo_event(
                        recording,
                        event,
                        delta,
                        &parameters,
                        &mut vo_timestamps,
                        &camera_matrices,
                        &mut frontend,
                        &mut stats,
                        &mut vo_only,
                        &mut has_pending_measurements,
                    )?;
                }
                EventKind::VisualOdometry(_) => {
                    stats.vo_received += 1;
                }
                EventKind::FieldMarkAssociations(associations)
                    if parameters.include_global_features && !recompute_global_features =>
                {
                    if let Some(anchor_time) = ingest_recorded_field_mark_associations(
                        &mut frontend,
                        event,
                        associations,
                        parameters.timestamp_mode,
                        parameters.pose_hint_visual_min_features_per_frame,
                        &mut stats,
                        &mut has_pending_measurements,
                    )? {
                        last_visual_anchor_time = Some(anchor_time);
                    }
                }
                EventKind::DetectedObjects(frame)
                    if parameters.include_global_features && recompute_global_features =>
                {
                    if let Some(anchor_time) = ingest_recomputed_global_features(
                        recording,
                        event,
                        frame,
                        &camera_matrices,
                        &mut frontend,
                        &mut association_state,
                        &parameters,
                        &field_dimensions,
                        &mut stats,
                        &mut has_pending_measurements,
                    )? {
                        last_visual_anchor_time = Some(anchor_time);
                    }
                }
                EventKind::DetectedObjects(_) | EventKind::FieldMarkAssociations(_) => {}
                _ => {}
            },
        }

        if index % 250 == 0 {
            let _ = sender.send(ResolveMessage::Progress(ResolveProgress {
                processed_events: index + 1,
                total_events,
                solve_count: samples.len(),
            }));
        }
    }

    if has_pending_measurements {
        solve_and_record(
            &mut backend,
            &mut frontend,
            recording,
            parameters.timestamp_mode,
            range_end_time,
            &mut stats,
            &parameters,
            last_visual_anchor_time,
            &mut last_accepted_robot_to_field,
            &mut samples,
        )?;
    }

    Ok(Some(ResolveResult {
        parameters,
        samples,
        vo_trajectory: vo_only.trajectory,
        stats,
        elapsed: started.elapsed(),
    }))
}

fn merged_replay_events<'a>(
    recording: &'a Recording,
    visual_odometry_override: Option<&'a [ReplayVisualOdometryEvent]>,
) -> Vec<ReplayEvent<'a>> {
    let override_visual_odometry = visual_odometry_override.is_some();
    let override_count = visual_odometry_override.map_or(0, <[ReplayVisualOdometryEvent]>::len);
    let mut replay_events = Vec::with_capacity(recording.event_count() + override_count);
    replay_events.extend(recording.events().iter().filter_map(|event| {
        if override_visual_odometry && matches!(event.kind, EventKind::VisualOdometry(_)) {
            None
        } else {
            Some(ReplayEvent::Recorded(event))
        }
    }));
    if let Some(visual_odometry_override) = visual_odometry_override {
        replay_events.extend(
            visual_odometry_override
                .iter()
                .map(ReplayEvent::VisualOdometryOverride),
        );
    }
    replay_events.sort_by_key(|event| nanos_since_epoch(event.log_time()));
    replay_events
}

fn backend_config(parameters: &ReplayParameters) -> BackendConfiguration {
    let localization_parameters = Localization3dParameters {
        visual_feature_noise_variance: parameters.visual_feature_noise_variance,
        pose_hint_visual_feature_noise_variance: parameters.pose_hint_visual_feature_noise_variance,
        pose_hint_visual_huber_threshold: parameters.pose_hint_visual_huber_threshold,
        reject_visual_odometry_during_head_motion: parameters
            .reject_visual_odometry_during_head_motion,
        max_visual_odometry_extrinsic_rotation: parameters.max_visual_odometry_extrinsic_rotation,
        max_visual_odometry_extrinsic_translation: parameters
            .max_visual_odometry_extrinsic_translation,
        require_recent_visual_anchor_for_large_pose_updates: parameters
            .require_recent_visual_anchor_for_large_pose_updates,
        max_visual_anchor_age: parameters.max_visual_anchor_age,
        max_unanchored_translation_update: parameters.max_unanchored_translation_update,
        max_unanchored_yaw_update: parameters.max_unanchored_yaw_update,
    };
    let mut config = backend_configuration_from_parameters(&localization_parameters);
    config.optimizer_max_iterations = parameters.optimizer_iterations.max(1);
    config.max_optimization_window =
        Duration::from_secs_f64(parameters.max_window_seconds.max(0.2));
    if parameters.override_visual_odometry_covariance {
        config.visual_odometry_noise =
            SMatrix::<f64, 6, 6>::identity() * parameters.visual_odometry_covariance.max(1.0e-12);
    }
    config.use_imu_kinematics = parameters.include_imu && parameters.include_imu_kinematics;
    config.use_imu_roll_pitch = parameters.include_imu && parameters.include_imu_roll_pitch;
    config.use_imu_yaw = parameters.include_imu && parameters.include_imu_yaw;
    config.use_current_spline_orientation =
        parameters.include_imu && parameters.include_current_spline_orientation;
    config
}

#[allow(clippy::too_many_arguments)]
fn ingest_vo_event(
    recording: &Recording,
    event: &RecordedEvent,
    delta: &VisualOdometryDelta,
    parameters: &ReplayParameters,
    vo_timestamps: &mut VisualOdometryTimestampTracker,
    camera_matrices: &OnlineCameraMatrices,
    frontend: &mut VinsFrontend,
    stats: &mut ReplayStats,
    vo_only: &mut VisualOdometryTrajectory,
    has_pending_measurements: &mut bool,
) -> Result<()> {
    stats.vo_received += 1;
    let timestamp_mode = parameters.timestamp_mode;
    let Some((previous_time, current_time)) =
        vo_timestamps.measurement_times(event, delta, timestamp_mode, recording)
    else {
        stats.vo_dropped_invalid += 1;
        return Ok(());
    };
    if current_time <= previous_time {
        stats.vo_dropped_invalid += 1;
        return Ok(());
    }

    let Some(previous_camera_matrix) = camera_matrices.nearest(previous_time, timestamp_mode)
    else {
        stats.vo_skipped_missing_camera_matrix += 1;
        return Ok(());
    };
    let Some(current_camera_matrix) = camera_matrices.nearest(current_time, timestamp_mode) else {
        stats.vo_skipped_missing_camera_matrix += 1;
        return Ok(());
    };
    if previous_camera_matrix.distance > CAMERA_MATRIX_MAX_TIME_DISTANCE
        || current_camera_matrix.distance > CAMERA_MATRIX_MAX_TIME_DISTANCE
    {
        stats.vo_skipped_stale_camera_matrix += 1;
        return Ok(());
    }
    if should_reject_visual_odometry_due_to_head_motion(
        visual_odometry_extrinsic_gate(parameters),
        &previous_camera_matrix.matrix.matrix.inner,
        &current_camera_matrix.matrix.matrix.inner,
    ) {
        stats.vo_skipped_head_motion += 1;
        return Ok(());
    }
    let current_robot_to_previous_robot = current_robot_to_previous_robot_from_visual_odometry(
        delta,
        &previous_camera_matrix.matrix.matrix.inner,
        &current_camera_matrix.matrix.matrix.inner,
    );
    vo_only.push(recording, current_time, current_robot_to_previous_robot);

    if visual_odometry_is_gated(
        delta,
        &previous_camera_matrix.matrix.matrix.inner,
        &current_camera_matrix.matrix.matrix.inner,
        parameters.max_visual_odometry_translation,
        parameters.max_visual_odometry_rotation,
    ) {
        stats.vo_dropped_gated += 1;
        return Ok(());
    }

    let mut delta = delta.clone();
    delta.previous_time = Time::from_wallclock(previous_time);
    delta.current_time = Time::from_wallclock(current_time);
    ingest_visual_odometry(
        frontend,
        delta,
        &previous_camera_matrix.matrix.matrix.inner,
        &current_camera_matrix.matrix.matrix.inner,
    )?;
    stats.vo_ingested += 1;
    *has_pending_measurements = true;
    Ok(())
}

fn visual_odometry_is_gated(
    delta: &VisualOdometryDelta,
    previous_camera_matrix: &CameraMatrix,
    current_camera_matrix: &CameraMatrix,
    max_translation: Option<f32>,
    max_rotation: Option<f32>,
) -> bool {
    if max_translation.is_none() && max_rotation.is_none() {
        return false;
    }

    let current_robot_to_previous_robot = current_robot_to_previous_robot_from_visual_odometry(
        delta,
        previous_camera_matrix,
        current_camera_matrix,
    );
    max_translation
        .is_some_and(|max| current_robot_to_previous_robot.translation.vector.norm() > max)
        || max_rotation.is_some_and(|max| current_robot_to_previous_robot.rotation.angle() > max)
}

struct VisualOdometryTrajectory {
    robot_to_field: nalgebra::Isometry3<f64>,
    trajectory: Vec<TrajectoryPoint>,
    segment_id: u64,
    previous_seconds: Option<f64>,
}

impl VisualOdometryTrajectory {
    fn new(robot_to_field: nalgebra::Isometry3<f64>) -> Self {
        Self {
            robot_to_field,
            trajectory: Vec::new(),
            segment_id: 0,
            previous_seconds: None,
        }
    }

    fn push(
        &mut self,
        recording: &Recording,
        time: SystemTime,
        current_robot_to_previous_robot: nalgebra::Isometry3<f32>,
    ) {
        self.robot_to_field = self.robot_to_field * current_robot_to_previous_robot.cast::<f64>();
        let seconds = recording.seconds_since_start(time);
        if let Some(previous_seconds) = self.previous_seconds
            && seconds - previous_seconds > TRAJECTORY_MAX_SAMPLE_GAP_SECONDS
        {
            self.segment_id += 1;
        }
        self.previous_seconds = Some(seconds);
        self.trajectory.push(TrajectoryPoint {
            seconds,
            robot_to_field: self.robot_to_field.framed_transform(),
            segment_id: self.segment_id,
        });
    }
}

fn current_robot_to_previous_robot_from_visual_odometry(
    delta: &VisualOdometryDelta,
    previous_camera_matrix: &CameraMatrix,
    current_camera_matrix: &CameraMatrix,
) -> nalgebra::Isometry3<f32> {
    let previous_robot_to_left_camera = robot_to_camera(previous_camera_matrix);
    let current_robot_to_left_camera = robot_to_camera(current_camera_matrix);
    previous_robot_to_left_camera.inverse()
        * delta.current_left_camera_to_previous_left_camera
        * current_robot_to_left_camera
}

fn visual_odometry_extrinsic_gate(parameters: &ReplayParameters) -> VisualOdometryExtrinsicGate {
    VisualOdometryExtrinsicGate {
        reject_during_head_motion: parameters.reject_visual_odometry_during_head_motion,
        max_rotation: parameters.max_visual_odometry_extrinsic_rotation,
        max_translation: parameters.max_visual_odometry_extrinsic_translation,
    }
}

fn visual_anchor_pose_gate(parameters: &ReplayParameters) -> VisualAnchorPoseGate {
    VisualAnchorPoseGate {
        require_recent_visual_anchor: parameters
            .require_recent_visual_anchor_for_large_pose_updates,
        max_anchor_age: parameters.max_visual_anchor_age,
        max_unanchored_translation_update: parameters.max_unanchored_translation_update as f64,
        max_unanchored_yaw_update: parameters.max_unanchored_yaw_update as f64,
    }
}

#[allow(clippy::too_many_arguments)]
fn ingest_recorded_field_mark_associations(
    frontend: &mut VinsFrontend,
    event: &RecordedEvent,
    recorded_associations: &TimeWrapper<FieldMarkAssociations>,
    timestamp_mode: TimestampMode,
    pose_hint_visual_min_features_per_frame: usize,
    stats: &mut ReplayStats,
    has_pending_measurements: &mut bool,
) -> Result<Option<SystemTime>> {
    stats.global_frames += 1;
    let associations = localization_visual_associations(
        recorded_associations.inner.associations.clone(),
        pose_hint_visual_min_features_per_frame,
    );
    if associations.is_empty() {
        return Ok(None);
    }

    stats.global_frames_ingested += 1;
    stats.global_associations_ingested += associations.len();
    record_ingested_association_stats(stats, &associations);
    let associations = associations
        .into_iter()
        .map(|association| VisualReprojectionAssociation {
            detection: association.detection,
            field_point: association.field_point,
            kind: match association.kind {
                FieldMarkAssociationKind::GlobalUnique => {
                    VisualReprojectionAssociationKind::GlobalUnique
                }
                FieldMarkAssociationKind::PoseHint => VisualReprojectionAssociationKind::PoseHint,
            },
        });
    let time = match timestamp_mode {
        TimestampMode::McapPublish => event.publish_time,
        TimestampMode::Embedded => recorded_associations.time.to_wallclock(),
    };
    frontend.ingest_visual_reprojection_associations(
        time,
        associations,
        recorded_associations.inner.robot_to_camera.inner,
    )?;
    *has_pending_measurements = true;
    Ok(Some(time))
}

fn localization_visual_associations(
    associations: Vec<FieldMarkAssociation>,
    pose_hint_visual_min_features_per_frame: usize,
) -> Vec<FieldMarkAssociation> {
    let has_global_association = associations
        .iter()
        .any(|association| association.kind == FieldMarkAssociationKind::GlobalUnique);
    if has_global_association {
        return associations
            .into_iter()
            .filter(|association| association.kind == FieldMarkAssociationKind::GlobalUnique)
            .collect();
    }
    if associations.len() >= pose_hint_visual_min_features_per_frame {
        associations
    } else {
        Vec::new()
    }
}

#[allow(clippy::too_many_arguments)]
fn ingest_recomputed_global_features(
    recording: &Recording,
    event: &RecordedEvent,
    frame: &crate::mcap_recording::DetectedObjectsFrame,
    camera_matrices: &OnlineCameraMatrices,
    frontend: &mut VinsFrontend,
    association_state: &mut FieldMarkAssociationState,
    parameters: &ReplayParameters,
    field_dimensions: &FieldDimensions,
    stats: &mut ReplayStats,
    has_pending_measurements: &mut bool,
) -> Result<Option<SystemTime>> {
    stats.global_frames += 1;
    let visual_features = find_detected_visual_features(&frame.objects);
    if visual_features.supported_feature_count() == 0 {
        return Ok(None);
    }
    stats.global_candidates += 1;
    let measurement_time =
        detected_objects_measurement_time(recording, event, frame, parameters.timestamp_mode);
    let Some(camera_matrix) = camera_matrices
        .nearest(measurement_time, parameters.timestamp_mode)
        .map(|nearest| nearest.matrix)
    else {
        return Ok(None);
    };
    if camera_matrix.distance_to(measurement_time, parameters.timestamp_mode)
        > CAMERA_MATRIX_MAX_TIME_DISTANCE
    {
        return Ok(None);
    }

    let pose_hint = frontend
        .peek_last_optimization_result()
        .filter(|result| {
            Duration::from_nanos(
                nanos_abs_diff(result.time, measurement_time).min(u64::MAX as u128) as u64,
            ) <= parameters.pose_hint.max_pose_age
        })
        .map(|result| {
            result
                .transform
                .cast::<f32>()
                .framed_transform::<Robot, Field>()
        });
    let association_parameters = FieldMarkAssociationParameters {
        global_localizer: parameters.global_localizer,
        pose_hint: parameters.pose_hint,
    };
    let localization = association_state.associate_visual_features_with_debug(
        &visual_features,
        &camera_matrix.matrix.inner,
        field_dimensions,
        pose_hint,
        &association_parameters,
        true,
    );
    match localization.debug.as_ref().map(|debug| debug.status) {
        None => stats.global_none += 1,
        Some(GlobalLocalizationDebugStatus::Ambiguous) => stats.global_ambiguous += 1,
        #[allow(deprecated)]
        Some(GlobalLocalizationDebugStatus::Unique) => stats.global_unique_modulo_symmetry += 1,
        Some(GlobalLocalizationDebugStatus::UniqueModuloSymmetry) => {
            stats.global_unique_modulo_symmetry += 1;
        }
    }

    let associations = localization_visual_associations(
        localization.associations,
        parameters.pose_hint_visual_min_features_per_frame,
    );
    if !associations.is_empty() {
        stats.global_frames_ingested += 1;
        stats.global_associations_ingested += associations.len();
        record_ingested_association_stats(stats, &associations);
        let associations =
            associations
                .into_iter()
                .map(|association| VisualReprojectionAssociation {
                    detection: association.detection,
                    field_point: association.field_point,
                    kind: match association.kind {
                        field_mark_association::FieldMarkAssociationKind::GlobalUnique => {
                            VisualReprojectionAssociationKind::GlobalUnique
                        }
                        field_mark_association::FieldMarkAssociationKind::PoseHint => {
                            VisualReprojectionAssociationKind::PoseHint
                        }
                    },
                });
        frontend.ingest_visual_reprojection_associations(
            measurement_time,
            associations,
            robot_to_camera(&camera_matrix.matrix.inner),
        )?;
        *has_pending_measurements = true;
        return Ok(Some(measurement_time));
    }
    Ok(None)
}

fn record_ingested_association_stats(
    stats: &mut ReplayStats,
    associations: &[FieldMarkAssociation],
) {
    let global_unique_count = associations
        .iter()
        .filter(|association| association.kind == FieldMarkAssociationKind::GlobalUnique)
        .count();
    let pose_hint_count = associations
        .iter()
        .filter(|association| association.kind == FieldMarkAssociationKind::PoseHint)
        .count();

    if global_unique_count > 0 {
        stats.global_unique_frames_ingested += 1;
        stats.global_unique_associations_ingested += global_unique_count;
    }
    if pose_hint_count > 0 {
        stats.pose_hint_frames_ingested += 1;
        stats.pose_hint_associations_ingested += pose_hint_count;
    }
}

fn detected_objects_measurement_time(
    recording: &Recording,
    event: &RecordedEvent,
    frame: &crate::mcap_recording::DetectedObjectsFrame,
    timestamp_mode: TimestampMode,
) -> SystemTime {
    match timestamp_mode {
        TimestampMode::McapPublish => recording
            .detected_objects_image_publish_time(event)
            .unwrap_or(event.publish_time),
        TimestampMode::Embedded => frame.display_time(),
    }
}

fn solve_and_record(
    backend: &mut VinsBackend,
    frontend: &mut VinsFrontend,
    recording: &Recording,
    timestamp_mode: TimestampMode,
    replay_time: SystemTime,
    stats: &mut ReplayStats,
    parameters: &ReplayParameters,
    last_visual_anchor_time: Option<SystemTime>,
    last_accepted_robot_to_field: &mut Option<nalgebra::Isometry3<f64>>,
    samples: &mut Vec<SolveSample>,
) -> Result<()> {
    let solve_started = Instant::now();
    let backend_result = backend.solve_once()?;
    let solve_duration = solve_started.elapsed();
    if backend_result.is_none() {
        return Ok(());
    }
    let Some(result) = frontend.last_optimization_result() else {
        return Ok(());
    };
    let mut robot_to_field = result.transform;
    if should_reject_unanchored_pose_update(
        visual_anchor_pose_gate(parameters),
        last_accepted_robot_to_field.as_ref(),
        &robot_to_field,
        last_visual_anchor_time,
        result.time,
    ) {
        if let Some(previous_robot_to_field) = last_accepted_robot_to_field {
            robot_to_field = *previous_robot_to_field;
            stats.pose_updates_reused_unanchored += 1;
        }
    } else {
        *last_accepted_robot_to_field = Some(robot_to_field);
    }

    samples.push(SolveSample {
        replay_seconds: recording.seconds_since_log_start(replay_time),
        graph_seconds: seconds_since(result.time, recording.graph_start_time(timestamp_mode)),
        solve_duration,
        robot_to_field: robot_to_field.framed_transform(),
        diagnostics: backend.compute_last_solve_diagnostics(),
        stats: stats.clone(),
    });
    Ok(())
}

#[derive(Default)]
struct OnlineCameraMatrices {
    matrices: Vec<OnlineCameraMatrix>,
}

impl OnlineCameraMatrices {
    fn push(&mut self, event: &RecordedEvent, matrix: TimeWrapper<CameraMatrix>) {
        self.matrices.push(OnlineCameraMatrix {
            publish_time: event.publish_time,
            matrix,
        });
    }

    fn nearest(
        &self,
        time: SystemTime,
        timestamp_mode: TimestampMode,
    ) -> Option<NearestCameraMatrix<'_>> {
        self.matrices
            .iter()
            .min_by_key(|candidate| nanos_abs_diff(candidate.time(timestamp_mode), time))
            .map(|matrix| NearestCameraMatrix {
                matrix,
                distance: Duration::from_nanos(
                    nanos_abs_diff(matrix.time(timestamp_mode), time).min(u64::MAX as u128) as u64,
                ),
            })
    }
}

struct OnlineCameraMatrix {
    publish_time: SystemTime,
    matrix: TimeWrapper<CameraMatrix>,
}

impl OnlineCameraMatrix {
    fn time(&self, timestamp_mode: TimestampMode) -> SystemTime {
        match timestamp_mode {
            TimestampMode::McapPublish => self.publish_time,
            TimestampMode::Embedded => self.matrix.time.to_wallclock(),
        }
    }

    fn distance_to(&self, time: SystemTime, timestamp_mode: TimestampMode) -> Duration {
        Duration::from_nanos(
            nanos_abs_diff(self.time(timestamp_mode), time).min(u64::MAX as u128) as u64,
        )
    }
}

fn robot_to_camera(camera_matrix: &CameraMatrix) -> nalgebra::Isometry3<f32> {
    (camera_matrix.head_to_camera * camera_matrix.robot_to_head).inner
}

struct NearestCameraMatrix<'a> {
    matrix: &'a OnlineCameraMatrix,
    distance: Duration,
}

#[derive(Default)]
struct VisualOdometryTimestampTracker;

impl VisualOdometryTimestampTracker {
    fn measurement_times(
        &mut self,
        _event: &RecordedEvent,
        delta: &VisualOdometryDelta,
        mode: TimestampMode,
        recording: &Recording,
    ) -> Option<(SystemTime, SystemTime)> {
        match mode {
            TimestampMode::Embedded => Some((
                delta.previous_time.to_wallclock(),
                delta.current_time.to_wallclock(),
            )),
            TimestampMode::McapPublish => {
                let previous = recording.aligned_image_time(delta.previous_time)?;
                let current = recording.aligned_image_time(delta.current_time)?;
                Some((previous, current))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trajectory_splits_segments_on_large_graph_gap() {
        let result = ResolveResult {
            parameters: ReplayParameters::default(),
            samples: vec![
                solve_sample_at(0.0),
                solve_sample_at(0.1),
                solve_sample_at(0.1 + TRAJECTORY_MAX_SAMPLE_GAP_SECONDS + 0.01),
            ],
            vo_trajectory: Vec::new(),
            stats: ReplayStats::default(),
            elapsed: Duration::ZERO,
        };

        let trajectory = result.trajectory();

        assert_eq!(trajectory.len(), 3);
        assert_eq!(trajectory[0].segment_id, 0);
        assert_eq!(trajectory[1].segment_id, 0);
        assert_eq!(trajectory[2].segment_id, 1);
    }

    fn solve_sample_at(graph_seconds: f64) -> SolveSample {
        SolveSample {
            replay_seconds: graph_seconds,
            graph_seconds,
            solve_duration: Duration::ZERO,
            robot_to_field: nalgebra::Isometry3::translation(graph_seconds, 0.0, 0.0)
                .framed_transform(),
            diagnostics: None,
            stats: ReplayStats::default(),
        }
    }
}
