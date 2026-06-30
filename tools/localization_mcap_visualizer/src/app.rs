use std::{
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver},
    },
    time::{Duration, Instant, SystemTime},
};

use color_eyre::{Result, eyre::eyre};
use coordinate_systems::{Field, Robot};
use eframe::{
    App, CreationContext, Frame,
    egui::{
        self, CentralPanel, Color32, ColorImage, Context, DragValue, FontId, Rect, RichText, Sense,
        SidePanel, Slider, Stroke, StrokeKind, TextureHandle, TextureOptions, TopBottomPanel, Ui,
        Vec2, Widget, pos2, vec2,
    },
};
use egui_bevy::BevyWidget;
use egui_plot::{Line, Plot, PlotPoints};
use field_mark_association::{
    FieldMarkAssociationKind, FieldMarkAssociations, GlobalLocalizationDebug,
    GlobalLocalizationDebugStatus, GlobalLocalizationDetailedDebug,
    GlobalLocalizationDetailedStatus, GlobalLocalizationScore, GlobalLocalizerParameters,
    VisualFeatureClass, find_detected_visual_features,
    localize_global_visual_features_detailed_debug,
};
use linear_algebra::IntoTransform;
use localization_3d::{SolveDiagnostics, initial_robot_to_field_from_camera_matrix};
use projection::camera_matrix::CameraMatrix;
use types::{
    field_dimensions::FieldDimensions,
    object_detection::{Object, RobocupObjectLabel},
    time_wrapper::TimeWrapper,
};

use crate::{
    mcap_recording::{
        CameraImage, Recording, SNAPSHOT_MAX_TIME_DISTANCE, StereoFrame, StereoImageId,
        TOPIC_ROBOT_KINEMATICS, TRAJECTORY_MAX_SAMPLE_GAP_SECONDS, TrajectoryPoint,
        nanos_since_epoch,
    },
    nearest_by_distance,
    replay::{ReplayParameters, ResolveMessage, ResolveProgress, ResolveResult, TimestampMode},
    scene::{
        self, SceneCameraFrame, SceneCameraSide, SceneData, SceneFrameSequence,
        SceneTrajectoryTrace, SceneVersion,
    },
};

pub struct LocalizationMcapVisualizerApp {
    recording: Arc<Recording>,
    mcap_path: PathBuf,
    widget: BevyWidget,
    position_seconds: f64,
    playing: bool,
    playback_rate: f64,
    last_frame_time: Instant,
    parameters: ReplayParameters,
    recorded_trajectory: Vec<TrajectoryPoint>,
    selected_camera: StereoSide,
    image_zoom: f32,
    image_cache: Option<CachedStereoFrame>,
    left_texture: Option<TextureHandle>,
    right_texture: Option<TextureHandle>,
    failed_image_id: Option<StereoImageId>,
    resolve: ResolveState,
    resolve_version: SceneVersion,
    camera_matrix_key: Option<CameraMatrixKey>,
    camera_version: SceneVersion,
    global_debug_cache: CachedGlobalDebug,
    use_global_debug_pose_in_3d: bool,
    project_robot_marker_to_ground: bool,
    show_top_down_path: bool,
    show_recorded_trajectory: bool,
    last_sent_show_recorded_trajectory: bool,
    show_vo_only: bool,
    last_sent_show_vo_only: bool,
    keep_trace_history: bool,
    show_trace_history: bool,
    trace_history: Vec<PathTrace>,
    trace_history_version: SceneVersion,
}

#[derive(Clone, Copy, Debug)]
enum ComponentPreset {
    Full,
    VisualOdometryOnly,
    GlobalFeaturesOnly,
    VisualAssociationsOnly,
    ImuRollPitchOnly,
    ImuYawOnly,
    ImuOrientationOnly,
    ImuKinematicsOnly,
    FootHeightsOnly,
    VisualOdometryAndImu,
    VisualOdometryAndFootHeights,
    VisualOdometryAndVisual,
}

#[derive(Clone, Debug)]
struct PathTrace {
    label: String,
    trajectory: Vec<TrajectoryPoint>,
    color: Color32,
}

impl LocalizationMcapVisualizerApp {
    pub fn new(
        creation_context: &CreationContext,
        mcap_path: PathBuf,
        recording: Arc<Recording>,
    ) -> Result<Self> {
        creation_context.egui_ctx.set_visuals(egui::Visuals::dark());

        let render_state = creation_context
            .wgpu_render_state
            .clone()
            .ok_or_else(|| eyre!("no WGPU render state found"))?;
        let mut widget = BevyWidget::new(render_state);
        scene::configure(&mut widget.bevy_app);
        widget.bevy_app.finish();
        widget.bevy_app.cleanup();

        Ok(Self {
            recorded_trajectory: recording.recorded_localization_trajectory(),
            recording,
            mcap_path,
            widget,
            position_seconds: 0.0,
            playing: false,
            playback_rate: 1.0,
            last_frame_time: Instant::now(),
            parameters: ReplayParameters::default(),
            selected_camera: StereoSide::Left,
            image_zoom: 1.0,
            image_cache: None,
            left_texture: None,
            right_texture: None,
            failed_image_id: None,
            resolve: ResolveState::Idle,
            resolve_version: SceneVersion::default(),
            camera_matrix_key: None,
            camera_version: SceneVersion::default(),
            global_debug_cache: CachedGlobalDebug::default(),
            use_global_debug_pose_in_3d: false,
            project_robot_marker_to_ground: true,
            show_top_down_path: true,
            show_recorded_trajectory: true,
            last_sent_show_recorded_trajectory: true,
            show_vo_only: false,
            last_sent_show_vo_only: false,
            keep_trace_history: true,
            show_trace_history: true,
            trace_history: Vec::new(),
            trace_history_version: SceneVersion::default(),
        })
    }
}

impl App for LocalizationMcapVisualizerApp {
    fn update(&mut self, context: &Context, _frame: &mut Frame) {
        self.advance_playback();
        self.poll_resolve();

        let display_time = self
            .recording
            .display_time_at_seconds(self.position_seconds);
        let snapshot = self.recording.latest_snapshot(display_time);
        self.update_stereo_textures(context, snapshot.image_id);

        let camera_matrix_key = snapshot
            .camera_matrix
            .as_ref()
            .map(|matrix| CameraMatrixKey {
                time_nanos: matrix.time.as_nanos(),
                calibrated_intrinsics_time_nanos: snapshot
                    .calibrated_intrinsics_time
                    .map(nanos_since_epoch),
            });
        let mut camera_matrix = snapshot.camera_matrix.clone().map(|mut matrix| {
            if let Some(intrinsics) = snapshot.calibrated_intrinsics {
                matrix.inner.intrinsics = intrinsics;
            }
            matrix.inner
        });
        if self.camera_matrix_key != camera_matrix_key {
            self.camera_matrix_key = camera_matrix_key;
            self.camera_version = self.camera_version.next();
        }

        let current_pose = self.current_robot_to_field(snapshot.recorded_localization);
        let debug_pose = self.robot_to_field_at_display_time(snapshot.detected_objects_time);
        let global_debug = self.update_global_debug_cache(
            camera_matrix.as_ref(),
            camera_matrix_key,
            snapshot.detected_objects_time,
            snapshot.global_localization_debug.as_ref(),
            snapshot.global_localization_debug_time,
            debug_pose,
            &snapshot.detected_objects,
        );
        let camera_matrix_for_ui = camera_matrix.clone();
        let scene_pose = if self.use_global_debug_pose_in_3d {
            global_debug
                .as_deref()
                .map(|debug| debug.robot_to_field.inner.cast().framed_transform())
                .unwrap_or(current_pose)
        } else {
            current_pose
        };
        self.update_scene_data(
            camera_matrix.take(),
            scene_pose,
            snapshot
                .robot_kinematics
                .as_ref()
                .map(|robot_kinematics| robot_kinematics.inner.clone()),
            global_debug.clone(),
        );

        let position_before_ui = self.position_seconds;
        let selected_camera_before_ui = self.selected_camera;
        let parameters_before_ui = self.parameters.clone();
        let resolve_version_before_ui = self.resolve_version;
        let playing_before_ui = self.playing;
        let use_global_debug_pose_before_ui = self.use_global_debug_pose_in_3d;
        let project_robot_marker_before_ui = self.project_robot_marker_to_ground;

        self.header(context);
        self.parameters_panel(context, snapshot.solve_diagnostics.as_ref());
        self.camera_panel(
            context,
            &snapshot.detected_objects,
            snapshot.detected_objects_time,
            snapshot.image_display_time,
            snapshot
                .field_mark_associations
                .as_ref()
                .map(|associations| &associations.inner),
            camera_matrix_for_ui.as_ref(),
            global_debug.as_deref(),
        );
        self.timeline_panel(context);
        self.viewport(context);
        let ui_changed_scene_inputs = self.position_seconds != position_before_ui
            || self.selected_camera != selected_camera_before_ui
            || self.parameters != parameters_before_ui
            || self.resolve_version != resolve_version_before_ui
            || self.playing != playing_before_ui
            || self.use_global_debug_pose_in_3d != use_global_debug_pose_before_ui
            || self.project_robot_marker_to_ground != project_robot_marker_before_ui;
        if self.playing
            || matches!(self.resolve, ResolveState::Running { .. })
            || ui_changed_scene_inputs
        {
            context.request_repaint();
        }
    }
}

impl LocalizationMcapVisualizerApp {
    fn advance_playback(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_frame_time).as_secs_f64();
        self.last_frame_time = now;
        if self.playing {
            self.position_seconds = (self.position_seconds + elapsed * self.playback_rate)
                .clamp(0.0, self.recording.duration().as_secs_f64());
            if self.position_seconds >= self.recording.duration().as_secs_f64() {
                self.playing = false;
            }
        }
    }

    fn poll_resolve(&mut self) {
        let mut finished = None;
        let mut failed = None;
        let mut cancelled = false;
        if let ResolveState::Running {
            receiver, progress, ..
        } = &mut self.resolve
        {
            while let Ok(message) = receiver.try_recv() {
                match message {
                    ResolveMessage::Progress(next_progress) => *progress = next_progress,
                    ResolveMessage::Finished(result) => finished = Some(result),
                    ResolveMessage::Failed(error) => failed = Some(error),
                    ResolveMessage::Cancelled => cancelled = true,
                }
            }
        }

        if let Some(result) = finished {
            if self.keep_trace_history {
                self.add_trace_from_result(&result);
            }
            self.resolve_version = self.resolve_version.next();
            self.resolve = ResolveState::Done(result);
        } else if let Some(error) = failed {
            self.resolve = ResolveState::Failed(error);
        } else if cancelled {
            self.resolve = ResolveState::Idle;
        }
    }

    fn update_scene_data(
        &mut self,
        camera_matrix: Option<CameraMatrix>,
        current_pose: linear_algebra::Isometry3<Robot, Field, f64>,
        robot_kinematics: Option<Arc<kinematics::robot_kinematics::RobotKinematics>>,
        global_debug: Option<Arc<GlobalLocalizationDetailedDebug>>,
    ) {
        let selected_frame_sequence = self.selected_scene_frame_sequence();
        let (
            current_frame_sequence,
            field_dimensions_version,
            camera_version,
            recorded_trajectory_version,
            resolved_trajectory_version,
            trace_trajectories_version,
            global_debug_version,
        ) = {
            let scene_data = self.widget.bevy_app.world_mut().resource::<SceneData>();
            (
                scene_data.camera_frame_sequence(),
                scene_data.field_dimensions_version(),
                scene_data.camera_version(),
                scene_data.recorded_trajectory_version(),
                scene_data.resolved_trajectory_version(),
                scene_data.trace_trajectories_version(),
                scene_data.global_debug_version(),
            )
        };
        let next_camera_frame = if current_frame_sequence != selected_frame_sequence {
            self.selected_scene_camera_frame()
        } else {
            None
        };
        let next_field_dimensions = if field_dimensions_version != SceneVersion::READY {
            Some(self.field_dimensions())
        } else {
            None
        };
        let trajectory_mode_changed = self.last_sent_show_vo_only != self.show_vo_only
            || self.last_sent_show_recorded_trajectory != self.show_recorded_trajectory;
        let next_recorded_trajectory =
            if recorded_trajectory_version != SceneVersion::READY || trajectory_mode_changed {
                Some(if self.show_vo_only || !self.show_recorded_trajectory {
                    Vec::new()
                } else {
                    self.recorded_trajectory.clone()
                })
            } else {
                None
            };
        let next_resolved_trajectory =
            if resolved_trajectory_version != self.resolve_version || trajectory_mode_changed {
                Some(match self.resolved_result() {
                    Some(result) if self.show_vo_only => result.vo_trajectory.clone(),
                    Some(result) => result.trajectory(),
                    None => Vec::new(),
                })
            } else {
                None
            };
        let next_trace_trajectories = if trace_trajectories_version != self.trace_history_version
            || trajectory_mode_changed
        {
            Some(self.scene_trace_trajectories())
        } else {
            None
        };

        let mut scene_data = self.widget.bevy_app.world_mut().resource_mut::<SceneData>();
        if let Some(field_dimensions) = next_field_dimensions {
            scene_data.set_field_dimensions(field_dimensions);
        }
        scene_data.set_current_robot_to_field(Some(current_pose.inner.cast().framed_transform()));
        scene_data.set_project_robot_marker_to_ground(self.project_robot_marker_to_ground);
        scene_data.set_robot_kinematics(robot_kinematics);

        if camera_version != self.camera_version {
            scene_data.set_camera_matrix(camera_matrix);
        }
        if current_frame_sequence != selected_frame_sequence {
            scene_data.set_camera_frame(next_camera_frame);
        }
        if let Some(recorded_trajectory) = next_recorded_trajectory {
            scene_data.set_recorded_trajectory(recorded_trajectory);
        }
        if let Some(resolved_trajectory) = next_resolved_trajectory {
            scene_data.set_resolved_trajectory(resolved_trajectory);
        }
        if let Some(trace_trajectories) = next_trace_trajectories {
            scene_data.set_trace_trajectories(trace_trajectories);
        }
        if global_debug_version != self.global_debug_cache.version {
            scene_data.set_global_debug(global_debug);
        }
        self.last_sent_show_recorded_trajectory = self.show_recorded_trajectory;
        self.last_sent_show_vo_only = self.show_vo_only;
    }

    fn selected_scene_frame_sequence(&self) -> Option<SceneFrameSequence> {
        self.image_cache
            .as_ref()
            .map(|cache| self.scene_frame_sequence_for(&cache.frame, self.scene_camera_side()))
    }

    fn selected_scene_camera_frame(&self) -> Option<SceneCameraFrame> {
        let side = self.scene_camera_side();
        self.image_cache.as_ref().map(|cache| SceneCameraFrame {
            sequence: self.scene_frame_sequence_for(&cache.frame, side),
            image: cache.active(self.scene_stereo_side(side)).clone(),
        })
    }

    fn scene_frame_sequence_for(
        &self,
        frame: &StereoFrame,
        side: SceneCameraSide,
    ) -> SceneFrameSequence {
        SceneFrameSequence::stereo(frame.sequence, side)
    }

    fn scene_camera_side(&self) -> SceneCameraSide {
        SceneCameraSide::Left
    }

    fn scene_stereo_side(&self, side: SceneCameraSide) -> StereoSide {
        match side {
            SceneCameraSide::Left => StereoSide::Left,
        }
    }

    fn update_stereo_textures(&mut self, context: &Context, image_id: Option<StereoImageId>) {
        let Some(image_id) = image_id else {
            self.image_cache = None;
            self.left_texture = None;
            self.right_texture = None;
            self.failed_image_id = None;
            return;
        };
        if self
            .image_cache
            .as_ref()
            .is_some_and(|cache| cache.image_id == image_id)
            || self.failed_image_id == Some(image_id)
        {
            return;
        }

        match self.recording.decode_stereo_image(image_id) {
            Ok(frame) => {
                self.set_texture(context, StereoSide::Left, &frame.left);
                self.set_texture(context, StereoSide::Right, &frame.right);
                self.image_cache = Some(CachedStereoFrame { image_id, frame });
                self.failed_image_id = None;
            }
            Err(error) => {
                eprintln!("failed to decode stereo frame {image_id}: {error:#}");
                self.image_cache = None;
                self.left_texture = None;
                self.right_texture = None;
                self.failed_image_id = Some(image_id);
            }
        }
    }

    fn set_texture(&mut self, context: &Context, side: StereoSide, image: &CameraImage) {
        let color_image = ColorImage::from_rgba_unmultiplied(
            [image.width as usize, image.height as usize],
            &image.rgba,
        );
        let texture = match side {
            StereoSide::Left => &mut self.left_texture,
            StereoSide::Right => &mut self.right_texture,
        };
        if let Some(texture) = texture {
            texture.set(color_image, TextureOptions::LINEAR);
        } else {
            *texture = Some(context.load_texture(
                match side {
                    StereoSide::Left => "localization_mcap_left_camera",
                    StereoSide::Right => "localization_mcap_right_camera",
                },
                color_image,
                TextureOptions::LINEAR,
            ));
        }
    }

    fn current_robot_to_field(
        &self,
        recorded_localization: Option<linear_algebra::Isometry3<Field, Robot>>,
    ) -> linear_algebra::Isometry3<Robot, Field, f64> {
        let pose = self
            .active_resolved_pose(self.position_seconds)
            .or_else(|| {
                recorded_localization
                    .map(|field_to_robot| field_to_robot.inverse().inner.cast().framed_transform())
            })
            .or_else(|| {
                nearest_trajectory_point(&self.recorded_trajectory, self.position_seconds)
                    .map(|point| point.robot_to_field)
            });
        match pose {
            Some(pose) => pose,
            None => self.initial_robot_to_field(),
        }
    }

    fn robot_to_field_at_display_time(
        &self,
        display_time: Option<SystemTime>,
    ) -> linear_algebra::Isometry3<Robot, Field, f64> {
        let seconds = match display_time {
            Some(display_time) => self.recording.seconds_since_start(display_time),
            None => self.position_seconds,
        };
        let pose = self.active_resolved_pose(seconds).or_else(|| {
            nearest_trajectory_point(&self.recorded_trajectory, seconds)
                .map(|point| point.robot_to_field)
        });
        match pose {
            Some(pose) => pose,
            None => self.initial_robot_to_field(),
        }
    }

    fn active_resolved_pose(
        &self,
        seconds: f64,
    ) -> Option<linear_algebra::Isometry3<Robot, Field, f64>> {
        let result = self.resolved_result()?;
        if self.show_vo_only {
            nearest_trajectory_point(&result.vo_trajectory, seconds)
                .map(|point| point.robot_to_field)
        } else {
            nearest_sample(&result.samples, seconds).map(|sample| sample.robot_to_field)
        }
    }

    fn active_resolved_trajectory(&self) -> Option<Vec<TrajectoryPoint>> {
        let result = self.resolved_result()?;
        Some(if self.show_vo_only {
            result.vo_trajectory.clone()
        } else {
            result.trajectory()
        })
    }

    fn scene_trace_trajectories(&self) -> Vec<SceneTrajectoryTrace> {
        if !self.show_trace_history {
            return Vec::new();
        }
        self.trace_history
            .iter()
            .map(|trace| SceneTrajectoryTrace {
                trajectory: trace.trajectory.clone(),
                color: color32_to_linear_rgba(trace.color),
            })
            .collect()
    }

    fn initial_robot_to_field(&self) -> linear_algebra::Isometry3<Robot, Field, f64> {
        initial_robot_to_field_from_camera_matrix(&self.recording.first_camera_matrix)
    }

    fn field_dimensions(&self) -> FieldDimensions {
        self.recording
            .field_dimensions
            .unwrap_or(FieldDimensions::SPL_2025)
    }

    fn update_global_debug_cache(
        &mut self,
        camera_matrix: Option<&projection::camera_matrix::CameraMatrix>,
        camera_matrix_key: Option<CameraMatrixKey>,
        detected_objects_time: Option<SystemTime>,
        recorded_global_debug: Option<&GlobalLocalizationDebug>,
        recorded_global_debug_time: Option<SystemTime>,
        current_pose: linear_algebra::Isometry3<Robot, Field, f64>,
        objects: &[Object<RobocupObjectLabel>],
    ) -> Option<Arc<GlobalLocalizationDetailedDebug>> {
        let key = if recorded_global_debug.is_some() || camera_matrix_key.is_some() {
            Some(GlobalDebugKey {
                camera_matrix_key,
                detected_objects_time_nanos: detected_objects_time.map(nanos_since_epoch),
                recorded_debug_time_nanos: recorded_global_debug_time.map(nanos_since_epoch),
                pose_revision: self.resolve_version,
                global_localizer: self.parameters.global_localizer,
            })
        } else {
            None
        };

        if self.global_debug_cache.key != key {
            let debug = recorded_global_debug
                .map(recorded_global_debug_to_detailed)
                .or_else(|| self.compute_global_debug(camera_matrix, current_pose, objects))
                .map(Arc::new);
            self.global_debug_cache.key = key;
            self.global_debug_cache.version = self.global_debug_cache.version.next();
            self.global_debug_cache.debug = debug;
        }

        self.global_debug_cache.debug.clone()
    }

    fn compute_global_debug(
        &self,
        camera_matrix: Option<&projection::camera_matrix::CameraMatrix>,
        current_pose: linear_algebra::Isometry3<Robot, Field, f64>,
        objects: &[Object<RobocupObjectLabel>],
    ) -> Option<GlobalLocalizationDetailedDebug> {
        let camera_matrix = camera_matrix?;
        let visual_features = find_detected_visual_features(objects);
        if visual_features.supported_feature_count()
            < self.parameters.global_localizer.min_inliers.max(3)
        {
            return None;
        }
        let pose_hint = Some(current_pose.inner.cast().framed_transform());
        let field_dimensions = self.field_dimensions();
        localize_global_visual_features_detailed_debug(
            &visual_features,
            camera_matrix,
            &field_dimensions,
            pose_hint,
            &self.parameters.global_localizer,
        )
    }

    fn header(&self, context: &Context) {
        TopBottomPanel::top("header").show(context, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.heading(RichText::new("Localization MCAP Visualizer").strong());
                ui.separator();
                ui.label(RichText::new(self.mcap_path.display().to_string()).monospace());
                ui.separator();
                ui.label(format!(
                    "{:.1}s, {} events, {} stereo frames, {} topics",
                    self.recording.duration().as_secs_f64(),
                    self.recording.event_count(),
                    self.recording.image_count(),
                    self.recording.topic_count(),
                ));
                ui.separator();
                ui.label(if self.recording.field_dimensions.is_some() {
                    "field: recorded field_dimensions"
                } else {
                    "field: SPL_2025 fallback"
                });
                let robot_kinematics_count =
                    self.recording.topic_message_count(TOPIC_ROBOT_KINEMATICS);
                if robot_kinematics_count > 0 {
                    ui.separator();
                    ui.label(format!("robot kinematics: {robot_kinematics_count}"));
                }
                if let Some(result) = self.resolved_result() {
                    ui.separator();
                    ui.colored_label(
                        Color32::LIGHT_GREEN,
                        format!(
                            "resolved: {} solves in {:.1}s",
                            result.samples.len(),
                            result.elapsed.as_secs_f64()
                        ),
                    );
                }
            });
        });
    }

    fn parameters_panel(
        &mut self,
        context: &Context,
        recorded_solve_diagnostics: Option<&TimeWrapper<SolveDiagnostics>>,
    ) {
        SidePanel::left("parameters_panel")
            .resizable(true)
            .default_width(360.0)
            .show(context, |ui| {
                ui.heading("Resolve Parameters");
                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    ui.label("timestamps");
                    ui.radio_value(
                        &mut self.parameters.timestamp_mode,
                        TimestampMode::McapPublish,
                        "MCAP/source",
                    );
                    ui.radio_value(
                        &mut self.parameters.timestamp_mode,
                        TimestampMode::Embedded,
                        "embedded",
                    );
                });
                numeric_row(
                    ui,
                    "solve cadence ms",
                    &mut self.parameters.solve_cadence_ms,
                    1.0..=500.0,
                );
                ui.horizontal(|ui| {
                    ui.label("optimizer iterations");
                    ui.add(DragValue::new(&mut self.parameters.optimizer_iterations).range(1..=50));
                });
                numeric_row(
                    ui,
                    "max window s",
                    &mut self.parameters.max_window_seconds,
                    0.2..=10.0,
                );
                let recording_duration = self.recording.duration().as_secs_f64();
                numeric_row(
                    ui,
                    "solve start s",
                    &mut self.parameters.solve_start_seconds,
                    0.0..=recording_duration,
                );
                numeric_row(
                    ui,
                    "solve end s",
                    &mut self.parameters.solve_end_seconds,
                    0.0..=recording_duration,
                );
                if !self.parameters.solve_end_seconds.is_finite()
                    || self.parameters.solve_end_seconds == 0.0
                {
                    self.parameters.solve_end_seconds = recording_duration;
                }
                if self.parameters.solve_start_seconds > self.parameters.solve_end_seconds {
                    self.parameters.solve_end_seconds = self.parameters.solve_start_seconds;
                }
                numeric_row(
                    ui,
                    "visual feature variance",
                    &mut self.parameters.visual_feature_noise_variance,
                    1.0..=100_000.0,
                );
                numeric_row(
                    ui,
                    "pose-hint visual variance",
                    &mut self.parameters.pose_hint_visual_feature_noise_variance,
                    1.0..=100_000.0,
                );
                numeric_row(
                    ui,
                    "pose-hint Huber",
                    &mut self.parameters.pose_hint_visual_huber_threshold,
                    0.1..=100.0,
                );
                ui.horizontal(|ui| {
                    ui.label("pose-hint min features");
                    ui.add(
                        DragValue::new(
                            &mut self.parameters.pose_hint_visual_min_features_per_frame,
                        )
                        .range(1..=16),
                    );
                });
                numeric_row(
                    ui,
                    "VO covariance",
                    &mut self.parameters.visual_odometry_covariance,
                    1.0e-8..=1.0,
                );
                ui.checkbox(
                    &mut self.parameters.override_visual_odometry_covariance,
                    "override VO covariance",
                );
                ui.checkbox(
                    &mut self.parameters.reject_visual_odometry_during_head_motion,
                    "reject VO during head motion",
                );
                ui.horizontal(|ui| {
                    ui.label("VO head rot rad");
                    ui.add(
                        DragValue::new(&mut self.parameters.max_visual_odometry_extrinsic_rotation)
                            .speed(0.001)
                            .range(0.0..=1.0),
                    );
                });
                ui.horizontal(|ui| {
                    ui.label("VO head trans m");
                    ui.add(
                        DragValue::new(
                            &mut self.parameters.max_visual_odometry_extrinsic_translation,
                        )
                        .speed(0.001)
                        .range(0.0..=0.2),
                    );
                });
                ui.checkbox(
                    &mut self
                        .parameters
                        .require_recent_visual_anchor_for_large_pose_updates,
                    "reuse pose for unanchored jumps",
                );
                ui.horizontal(|ui| {
                    ui.label("visual anchor age ms");
                    let mut max_visual_anchor_age_ms =
                        self.parameters.max_visual_anchor_age.as_secs_f64() * 1000.0;
                    if ui
                        .add(
                            DragValue::new(&mut max_visual_anchor_age_ms)
                                .speed(10.0)
                                .range(10.0..=5000.0),
                        )
                        .changed()
                    {
                        self.parameters.max_visual_anchor_age =
                            Duration::from_secs_f64(max_visual_anchor_age_ms / 1000.0);
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("unanchored xy m");
                    ui.add(
                        DragValue::new(&mut self.parameters.max_unanchored_translation_update)
                            .speed(0.01)
                            .range(0.0..=2.0),
                    );
                });
                ui.horizontal(|ui| {
                    ui.label("unanchored yaw rad");
                    ui.add(
                        DragValue::new(&mut self.parameters.max_unanchored_yaw_update)
                            .speed(0.01)
                            .range(0.0..=3.2),
                    );
                });
                ui.separator();
                ui.label(RichText::new("Component Presets").strong());
                ui.horizontal_wrapped(|ui| {
                    if ui.button("Full").clicked() {
                        self.apply_component_preset(ComponentPreset::Full);
                    }
                    if ui.button("VO only").clicked() {
                        self.apply_component_preset(ComponentPreset::VisualOdometryOnly);
                    }
                    if ui.button("Global only").clicked() {
                        self.apply_component_preset(ComponentPreset::GlobalFeaturesOnly);
                    }
                    if ui.button("Visual assoc").clicked() {
                        self.apply_component_preset(ComponentPreset::VisualAssociationsOnly);
                    }
                    if ui.button("IMU R/P").clicked() {
                        self.apply_component_preset(ComponentPreset::ImuRollPitchOnly);
                    }
                    if ui.button("IMU yaw").clicked() {
                        self.apply_component_preset(ComponentPreset::ImuYawOnly);
                    }
                    if ui.button("IMU RPY").clicked() {
                        self.apply_component_preset(ComponentPreset::ImuOrientationOnly);
                    }
                    if ui.button("IMU kin").clicked() {
                        self.apply_component_preset(ComponentPreset::ImuKinematicsOnly);
                    }
                    if ui.button("Foot").clicked() {
                        self.apply_component_preset(ComponentPreset::FootHeightsOnly);
                    }
                    if ui.button("VO+IMU").clicked() {
                        self.apply_component_preset(ComponentPreset::VisualOdometryAndImu);
                    }
                    if ui.button("VO+Foot").clicked() {
                        self.apply_component_preset(ComponentPreset::VisualOdometryAndFootHeights);
                    }
                    if ui.button("VO+Visual").clicked() {
                        self.apply_component_preset(ComponentPreset::VisualOdometryAndVisual);
                    }
                });
                ui.separator();
                ui.checkbox(
                    &mut self.parameters.include_visual_odometry,
                    "include visual odometry",
                );
                ui.checkbox(
                    &mut self.parameters.include_global_features,
                    "include global visual features",
                );
                ui.checkbox(
                    &mut self.parameters.recompute_global_features,
                    "recompute global features from detections",
                );
                ui.checkbox(&mut self.parameters.include_imu, "include IMU");
                ui.indent("imu_component_toggles", |ui| {
                    ui.checkbox(
                        &mut self.parameters.include_imu_roll_pitch,
                        "roll/pitch attitude",
                    );
                    ui.checkbox(&mut self.parameters.include_imu_yaw, "relative yaw");
                    ui.checkbox(
                        &mut self.parameters.include_current_spline_orientation,
                        "current RPY inside active interval",
                    );
                    ui.checkbox(
                        &mut self.parameters.include_imu_kinematics,
                        "gyro/accel kinematics",
                    );
                });
                ui.checkbox(
                    &mut self.parameters.include_foot_heights,
                    "include foot heights",
                );
                ui.separator();
                ui.label(RichText::new("Global Localizer").strong());
                ui.horizontal(|ui| {
                    ui.label("min inliers");
                    ui.add(
                        DragValue::new(&mut self.parameters.global_localizer.min_inliers)
                            .range(3..=16),
                    );
                });
                ui.horizontal(|ui| {
                    ui.label("min confidence");
                    ui.add(
                        DragValue::new(&mut self.parameters.global_localizer.min_confidence)
                            .speed(0.01)
                            .range(0.0..=1.0),
                    );
                });
                ui.horizontal(|ui| {
                    ui.label("detection baseline");
                    ui.add(
                        DragValue::new(
                            &mut self.parameters.global_localizer.min_detection_baseline,
                        )
                        .speed(0.01)
                        .range(0.01..=2.0),
                    );
                });
                ui.horizontal(|ui| {
                    ui.label("map baseline m");
                    ui.add(
                        DragValue::new(&mut self.parameters.global_localizer.min_map_baseline)
                            .speed(0.01)
                            .range(0.01..=2.0),
                    );
                });
                ui.horizontal(|ui| {
                    ui.label("height min/max");
                    ui.add(
                        DragValue::new(&mut self.parameters.global_localizer.height_min)
                            .speed(0.01)
                            .range(0.05..=2.0),
                    );
                    ui.add(
                        DragValue::new(&mut self.parameters.global_localizer.height_max)
                            .speed(0.01)
                            .range(0.05..=2.0),
                    );
                });
                if self.parameters.global_localizer.height_min
                    > self.parameters.global_localizer.height_max
                {
                    self.parameters.global_localizer.height_max =
                        self.parameters.global_localizer.height_min;
                }
                ui.horizontal(|ui| {
                    ui.label("association gate m");
                    ui.add(
                        DragValue::new(&mut self.parameters.global_localizer.association_gate)
                            .speed(0.01)
                            .range(0.05..=2.0),
                    );
                });
                ui.horizontal(|ui| {
                    ui.label("RMS threshold m");
                    ui.add(
                        DragValue::new(&mut self.parameters.global_localizer.rms_threshold)
                            .speed(0.01)
                            .range(0.01..=2.0),
                    );
                });
                ui.horizontal(|ui| {
                    ui.label("min score");
                    ui.add(
                        DragValue::new(&mut self.parameters.global_localizer.min_score)
                            .speed(0.01)
                            .range(0.0..=10.0),
                    );
                });
                ui.horizontal(|ui| {
                    ui.label("score ratio");
                    ui.add(
                        DragValue::new(&mut self.parameters.global_localizer.score_ratio)
                            .speed(0.01)
                            .range(1.0..=10.0),
                    );
                });
                ui.horizontal(|ui| {
                    ui.label("residual weight");
                    ui.add(
                        DragValue::new(&mut self.parameters.global_localizer.residual_weight)
                            .speed(0.01)
                            .range(0.0..=10.0),
                    );
                });
                ui.separator();
                ui.label(RichText::new("Pose-Hint Fallback").strong());
                ui.checkbox(&mut self.parameters.pose_hint.enabled, "enabled");
                ui.horizontal(|ui| {
                    ui.label("max pose age ms");
                    let mut max_pose_age_ms =
                        self.parameters.pose_hint.max_pose_age.as_secs_f64() * 1000.0;
                    if ui
                        .add(
                            DragValue::new(&mut max_pose_age_ms)
                                .speed(1.0)
                                .range(1.0..=5000.0),
                        )
                        .changed()
                    {
                        self.parameters.pose_hint.max_pose_age =
                            Duration::from_secs_f64(max_pose_age_ms / 1000.0);
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("reprojection gate px");
                    ui.add(
                        DragValue::new(&mut self.parameters.pose_hint.max_reprojection_error_px)
                            .speed(1.0)
                            .range(1.0..=500.0),
                    );
                });
                ui.horizontal(|ui| {
                    ui.label("second-best margin px");
                    ui.add(
                        DragValue::new(
                            &mut self.parameters.pose_hint.second_best_reprojection_margin_px,
                        )
                        .speed(1.0)
                        .range(0.0..=200.0),
                    );
                });
                ui.horizontal(|ui| {
                    ui.label("healthy min inliers");
                    ui.add(
                        DragValue::new(&mut self.parameters.pose_hint.healthy_min_inliers)
                            .speed(1.0)
                            .range(1..=20),
                    );
                });
                ui.horizontal(|ui| {
                    ui.label("healthy RMSE px");
                    ui.add(
                        DragValue::new(&mut self.parameters.pose_hint.healthy_max_rmse_px)
                            .speed(1.0)
                            .range(1.0..=200.0),
                    );
                });
                ui.horizontal(|ui| {
                    ui.label("recovery frames");
                    ui.add(
                        DragValue::new(&mut self.parameters.pose_hint.recovery_frames)
                            .speed(1.0)
                            .range(1..=30),
                    );
                });
                ui.horizontal(|ui| {
                    ui.label("recovery distance m");
                    ui.add(
                        DragValue::new(&mut self.parameters.pose_hint.recovery_max_pose_distance)
                            .speed(0.05)
                            .range(0.05..=5.0),
                    );
                });
                ui.horizontal(|ui| {
                    ui.label("recovery yaw rad");
                    ui.add(
                        DragValue::new(&mut self.parameters.pose_hint.recovery_max_pose_angle)
                            .speed(0.01)
                            .range(0.01..=std::f32::consts::PI),
                    );
                });
                ui.separator();
                self.resolve_controls(ui);
                ui.separator();
                self.diagnostics(ui, recorded_solve_diagnostics);
            });
    }

    fn apply_component_preset(&mut self, preset: ComponentPreset) {
        let pose_hint_enabled = matches!(
            preset,
            ComponentPreset::Full | ComponentPreset::VisualAssociationsOnly
        );
        self.parameters.include_visual_odometry = matches!(
            preset,
            ComponentPreset::Full
                | ComponentPreset::VisualOdometryOnly
                | ComponentPreset::VisualOdometryAndImu
                | ComponentPreset::VisualOdometryAndFootHeights
                | ComponentPreset::VisualOdometryAndVisual
        );
        self.parameters.include_global_features = matches!(
            preset,
            ComponentPreset::Full
                | ComponentPreset::GlobalFeaturesOnly
                | ComponentPreset::VisualAssociationsOnly
                | ComponentPreset::VisualOdometryAndVisual
        );
        self.parameters.include_foot_heights = matches!(
            preset,
            ComponentPreset::Full
                | ComponentPreset::FootHeightsOnly
                | ComponentPreset::VisualOdometryAndFootHeights
        );

        self.parameters.include_imu_roll_pitch = matches!(
            preset,
            ComponentPreset::Full
                | ComponentPreset::ImuRollPitchOnly
                | ComponentPreset::ImuOrientationOnly
                | ComponentPreset::VisualOdometryAndImu
        );
        self.parameters.include_imu_yaw = matches!(
            preset,
            ComponentPreset::Full
                | ComponentPreset::ImuYawOnly
                | ComponentPreset::ImuOrientationOnly
                | ComponentPreset::VisualOdometryAndImu
        );
        self.parameters.include_current_spline_orientation = matches!(
            preset,
            ComponentPreset::Full
                | ComponentPreset::ImuOrientationOnly
                | ComponentPreset::VisualOdometryAndImu
        );
        self.parameters.include_imu_kinematics = matches!(
            preset,
            ComponentPreset::Full
                | ComponentPreset::ImuKinematicsOnly
                | ComponentPreset::VisualOdometryAndImu
        );
        self.parameters.include_imu = self.parameters.include_imu_roll_pitch
            || self.parameters.include_imu_yaw
            || self.parameters.include_current_spline_orientation
            || self.parameters.include_imu_kinematics;
        self.parameters.pose_hint.enabled = pose_hint_enabled;
    }

    fn add_trace_from_result(&mut self, result: &ResolveResult) {
        let trajectory = if self.show_vo_only {
            result.vo_trajectory.clone()
        } else {
            result.trajectory()
        };
        if trajectory.len() < 2 {
            return;
        }

        let color = trace_color(self.trace_history.len());
        self.trace_history.push(PathTrace {
            label: component_trace_label(&result.parameters, self.show_vo_only),
            trajectory,
            color,
        });
        self.trace_history_version = self.trace_history_version.next();
    }

    fn resolve_controls(&mut self, ui: &mut Ui) {
        match &mut self.resolve {
            ResolveState::Running {
                cancel, progress, ..
            } => {
                ui.horizontal(|ui| {
                    if ui.button("Cancel").clicked() {
                        cancel.store(true, Ordering::Relaxed);
                    }
                    ui.label(format!(
                        "{} / {} events, {} solves",
                        progress.processed_events, progress.total_events, progress.solve_count
                    ));
                });
                if progress.total_events > 0 {
                    ui.add(
                        egui::ProgressBar::new(
                            progress.processed_events as f32 / progress.total_events as f32,
                        )
                        .show_percentage(),
                    );
                }
            }
            _ => {
                if ui.button(RichText::new("Resolve").strong()).clicked() {
                    let (sender, receiver) = mpsc::channel();
                    let cancel = Arc::new(AtomicBool::new(false));
                    crate::replay::spawn_resolve(
                        self.recording.clone(),
                        self.parameters.clone(),
                        cancel.clone(),
                        sender,
                    );
                    self.resolve_version = self.resolve_version.next();
                    self.resolve = ResolveState::Running {
                        receiver,
                        cancel,
                        progress: ResolveProgress {
                            processed_events: 0,
                            total_events: self.recording.event_count(),
                            solve_count: 0,
                        },
                    };
                }
                if self.resolved_result().is_some() {
                    let label = if self.show_vo_only {
                        "Show solver"
                    } else {
                        "Show VO only"
                    };
                    if ui.button(label).clicked() {
                        self.show_vo_only = !self.show_vo_only;
                        self.resolve_version = self.resolve_version.next();
                    }
                }
                ui.horizontal_wrapped(|ui| {
                    ui.checkbox(
                        &mut self.use_global_debug_pose_in_3d,
                        "use global pose in 3D",
                    );
                    ui.checkbox(
                        &mut self.project_robot_marker_to_ground,
                        "ground-yaw robot marker",
                    );
                    let recorded_changed = ui
                        .checkbox(&mut self.show_recorded_trajectory, "show recorded path")
                        .changed();
                    if recorded_changed {
                        self.resolve_version = self.resolve_version.next();
                    }
                    ui.checkbox(&mut self.keep_trace_history, "keep displayed paths");
                    let show_changed = ui
                        .checkbox(&mut self.show_trace_history, "show kept paths")
                        .changed();
                    if show_changed {
                        self.trace_history_version = self.trace_history_version.next();
                    }
                    if ui.button("Pin displayed path").clicked() {
                        if let Some(result) = self.resolved_result().cloned() {
                            self.add_trace_from_result(&result);
                        }
                    }
                    if ui.button("Clear paths").clicked() {
                        self.trace_history.clear();
                        self.trace_history_version = self.trace_history_version.next();
                    }
                    if ui.button("Print solver samples").clicked() {
                        if let Some(result) = self.resolved_result() {
                            print_solver_samples(result);
                        }
                    }
                });
                if let ResolveState::Failed(error) = &self.resolve {
                    ui.colored_label(Color32::LIGHT_RED, error);
                }
            }
        }
    }

    fn diagnostics(
        &self,
        ui: &mut Ui,
        recorded_solve_diagnostics: Option<&TimeWrapper<SolveDiagnostics>>,
    ) {
        ui.heading("Diagnostics");
        if let Some(diagnostics) = recorded_solve_diagnostics {
            ui.label(format!(
                "recorded solve diagnostics at {:.2}s",
                self.recording
                    .seconds_since_start(diagnostics.time.to_wallclock())
            ));
            recorded_solve_diagnostics_summary(ui, &diagnostics.inner);
            ui.separator();
        }

        let Some(result) = self.resolved_result() else {
            ui.label(
                RichText::new("Run Resolve to populate replay solve diagnostics.")
                    .color(Color32::GRAY),
            );
            return;
        };

        let stats = &result.stats;
        ui.label(if self.show_vo_only {
            RichText::new("displaying VO-only trajectory").color(Color32::LIGHT_GREEN)
        } else {
            RichText::new("displaying solver trajectory").color(Color32::GRAY)
        });
        ui.label(format!(
            "VO: {} received, {} ingested, {} head-motion skips, {} stale camera skips",
            stats.vo_received,
            stats.vo_ingested,
            stats.vo_skipped_head_motion,
            stats.vo_skipped_stale_camera_matrix
        ));
        ui.label(format!(
            "IMU/foot: {} IMU messages, {} foot-height frames ingested",
            stats.imu_ingested, stats.foot_heights_ingested
        ));
        ui.label(format!(
            "Pose: {} unanchored jumps reused previous pose",
            stats.pose_updates_reused_unanchored
        ));
        ui.label(format!(
            "Global: {} frames, {} candidates, {} ingested, {} associations",
            stats.global_frames,
            stats.global_candidates,
            stats.global_frames_ingested,
            stats.global_associations_ingested
        ));
        if let Some(sample) = nearest_sample(&result.samples, self.position_seconds) {
            ui.separator();
            ui.label(format!(
                "selected solve: {:.2} ms",
                sample.solve_duration.as_secs_f64() * 1000.0
            ));
            ui.label(format!("graph time: {:.2}s", sample.graph_seconds));
            ui.label(format!("replay time: {:.2}s", sample.replay_seconds));
            ui.label(format!(
                "cumulative VO/head skips/reused/global: {} / {} / {} / {}",
                sample.stats.vo_ingested,
                sample.stats.vo_skipped_head_motion,
                sample.stats.pose_updates_reused_unanchored,
                sample.stats.global_associations_ingested
            ));
            if let Some(diagnostics) = &sample.diagnostics {
                ui.label(format!("optimizer: {:?}", diagnostics.optimizer_status));
                ui.label(format!(
                    "values/factors: {} / {}",
                    diagnostics.value_count, diagnostics.factor_count
                ));
                ui.label(format!("total error: {:.3}", diagnostics.total_error));
                ui.label(format!(
                    "VO RMS mean/max: {:.3} / {:.3}",
                    diagnostics.visual_odometry.mean_rms, diagnostics.visual_odometry.max_rms
                ));
                ui.label(format!(
                    "visual RMS mean/max: {:.3} / {:.3}",
                    diagnostics.visual_reprojection.mean_rms,
                    diagnostics.visual_reprojection.max_rms
                ));
                ui.label(format!(
                    "foot RMS mean/max: {:.3} / {:.3}",
                    diagnostics.foot_above_ground.mean_rms,
                    diagnostics.foot_above_ground.max_rms
                ));
                ui.label(format!(
                    "GP RMS mean/max: {:.3} / {:.3}",
                    diagnostics.gaussian_process_prior.mean_rms,
                    diagnostics.gaussian_process_prior.max_rms
                ));
            }
        }
        Plot::new("solve_duration_plot")
            .height(140.0)
            .show(ui, |plot_ui| {
                let points = PlotPoints::from_iter(result.samples.iter().map(|sample| {
                    [
                        sample.graph_seconds,
                        sample.solve_duration.as_secs_f64() * 1000.0,
                    ]
                }));
                plot_ui.line(Line::new("solve ms", points));
            });
        ui.label(format!(
            "resolved with {} iterations, {:.1} ms cadence, visual variance {:.1}",
            result.parameters.optimizer_iterations,
            result.parameters.solve_cadence_ms,
            result.parameters.visual_feature_noise_variance,
        ));
        ui.label(format!(
            "components: VO={} visual={} pose-hint={} IMU[rp={}, yaw={}, kin={}] foot={}",
            result.parameters.include_visual_odometry,
            result.parameters.include_global_features,
            result.parameters.pose_hint.enabled,
            result.parameters.include_imu && result.parameters.include_imu_roll_pitch,
            result.parameters.include_imu && result.parameters.include_imu_yaw,
            result.parameters.include_imu && result.parameters.include_imu_kinematics,
            result.parameters.include_foot_heights,
        ));
        if self.show_trace_history && !self.trace_history.is_empty() {
            ui.separator();
            ui.label(RichText::new("Kept paths").strong());
            for trace in &self.trace_history {
                ui.colored_label(trace.color, &trace.label);
            }
        }
    }

    fn camera_panel(
        &mut self,
        context: &Context,
        detected_objects: &[Object<RobocupObjectLabel>],
        detected_objects_time: Option<SystemTime>,
        image_display_time: Option<SystemTime>,
        field_mark_associations: Option<&FieldMarkAssociations>,
        camera_matrix: Option<&CameraMatrix>,
        global_debug: Option<&GlobalLocalizationDetailedDebug>,
    ) {
        SidePanel::right("camera_panel")
            .resizable(true)
            .default_width(520.0)
            .show(context, |ui| {
                ui.horizontal(|ui| {
                    ui.heading("Stereo Camera");
                    ui.selectable_value(&mut self.selected_camera, StereoSide::Left, "left");
                    ui.selectable_value(&mut self.selected_camera, StereoSide::Right, "right");
                });
                ui.horizontal(|ui| {
                    ui.label("zoom");
                    ui.add(Slider::new(&mut self.image_zoom, 0.1..=8.0).logarithmic(true));
                    if ui.button("1:1").clicked() {
                        self.image_zoom = 1.0;
                    }
                });
                let texture = match self.selected_camera {
                    StereoSide::Left => self.left_texture.as_ref(),
                    StereoSide::Right => self.right_texture.as_ref(),
                };
                let image = self
                    .image_cache
                    .as_ref()
                    .map(|cache| cache.active(self.selected_camera));
                match (texture, image) {
                    (Some(texture), Some(image)) => self.camera_image(
                        ui,
                        texture,
                        image,
                        detected_objects,
                        detected_objects_time,
                        image_display_time,
                        field_mark_associations,
                        global_debug,
                    ),
                    _ => {
                        ui.centered_and_justified(|ui| {
                            ui.label(
                                RichText::new(format!(
                                    "no stereo image within {:.0} ms of this display time",
                                    SNAPSHOT_MAX_TIME_DISTANCE.as_secs_f64() * 1000.0
                                ))
                                .color(Color32::GRAY),
                            );
                        });
                    }
                }
                ui.separator();
                self.top_down_association_view(
                    ui,
                    global_debug,
                    camera_matrix,
                    detected_objects_time,
                );
                ui.separator();
                self.global_debug_panel(ui, global_debug);
            });
    }

    fn camera_image(
        &self,
        ui: &mut Ui,
        texture: &TextureHandle,
        image: &CameraImage,
        detected_objects: &[Object<RobocupObjectLabel>],
        detected_objects_time: Option<SystemTime>,
        image_display_time: Option<SystemTime>,
        field_mark_associations: Option<&FieldMarkAssociations>,
        global_debug: Option<&GlobalLocalizationDetailedDebug>,
    ) {
        let image_size = vec2(image.width as f32, image.height as f32);
        let available = ui.available_size().max(vec2(1.0, 1.0));
        let fit_scale = (available.x / image_size.x)
            .min((available.y - 300.0).max(1.0) / image_size.y)
            .max(0.05);
        let scale = fit_scale * self.image_zoom;
        egui::ScrollArea::both()
            .max_height((available.y - 280.0).max(220.0))
            .show(ui, |ui| {
                let response = ui.add(
                    egui::Image::new((texture.id(), texture.size_vec2()))
                        .fit_to_exact_size(image_size * scale)
                        .sense(Sense::hover()),
                );
                if self.selected_camera == StereoSide::Left {
                    draw_detected_objects(ui, response.rect, image_size, detected_objects);
                    if let Some(field_mark_associations) = field_mark_associations {
                        draw_recorded_association_pixels(
                            ui,
                            response.rect,
                            image_size,
                            field_mark_associations,
                        );
                    }
                    if let Some(debug) = global_debug {
                        draw_global_debug_overlay(ui, response.rect, image_size, debug);
                    }
                    let hover_info =
                        image_hover_info(&response, image_size, detected_objects, global_debug);
                    if !hover_info.is_empty() {
                        response.on_hover_ui(|ui| {
                            for line in hover_info {
                                ui.label(line);
                            }
                        });
                    }
                }
            });
        if self.selected_camera == StereoSide::Right {
            ui.colored_label(
                Color32::GRAY,
                "detection and global-localization overlays are left-camera only",
            );
        }
        ui.horizontal_wrapped(|ui| {
            ui.label(format!("{}x{}", image.width, image.height));
            ui.separator();
            ui.label(format!("{} left detections", detected_objects.len()));
            if let Some(field_mark_associations) = field_mark_associations {
                ui.separator();
                ui.label(format!(
                    "{} recorded associations",
                    field_mark_associations.associations.len()
                ));
            }
            if let Some(cache) = &self.image_cache {
                let frame_display_time =
                    image_display_time.unwrap_or_else(|| cache.frame.source_time.to_wallclock());
                ui.separator();
                ui.label(format!("frame {}", cache.image_id));
                ui.separator();
                ui.label(format!(
                    "image {:.2}s source {:?}",
                    self.recording.seconds_since_start(frame_display_time),
                    cache.frame.source_time,
                ));
                ui.separator();
                ui.label(format!(
                    "log {:.2}s publish {:.2}s",
                    self.recording.seconds_since_log_start(cache.frame.log_time),
                    self.recording
                        .seconds_since_log_start(cache.frame.publish_time),
                ));
                if let Some(detected_objects_time) = detected_objects_time {
                    let delta_ms = (self.recording.seconds_since_start(detected_objects_time)
                        - self.recording.seconds_since_start(frame_display_time))
                        * 1000.0;
                    ui.separator();
                    ui.label(format!("detections Δ {delta_ms:.1} ms"));
                }
            }
        });
    }

    fn global_debug_panel(
        &self,
        ui: &mut Ui,
        global_debug: Option<&GlobalLocalizationDetailedDebug>,
    ) {
        ui.heading("Global Localization Debug");
        let Some(debug) = global_debug else {
            ui.label(
                RichText::new("No global-localization candidate at this frame.")
                    .color(Color32::GRAY),
            );
            return;
        };
        ui.label(format!("status: {:?}", debug.status));
        ui.label(format!("inliers: {}", debug.score.inliers));
        ui.label(format!(
            "candidate score: {:.3}",
            debug.score.candidate_score
        ));
        ui.label(format!(
            "metric RMS: {:.3}m",
            debug.score.metric_rms_residual
        ));
        ui.label(format!("RMSE: {:.2}px", debug.score.reprojection_rmse));
        ui.label(format!("total cost: {:.1}", debug.score.total_cost));
        ui.label(format!(
            "detections: {}, projected candidates: {}, associations: {}",
            debug.detections.len(),
            debug.projected_features.len(),
            debug.associations.len()
        ));
        if debug.detections.is_empty() && debug.projected_features.is_empty() {
            ui.label(
                RichText::new(
                    "Recorded debug summary only; detailed projections were not recorded.",
                )
                .color(Color32::GRAY),
            );
        }
        ui.separator();
        egui::ScrollArea::vertical()
            .max_height(180.0)
            .show(ui, |ui| {
                for association in &debug.associations {
                    let error = match association.reprojection_error_px {
                        Some(error) => format!("{error:.1}px"),
                        None => "not visible".to_string(),
                    };
                    ui.label(format!(
                        "#{}/#{} {:?}: det ({:.1},{:.1}) -> field ({:.2},{:.2}) error {error}",
                        association.detection_index,
                        association.feature_index,
                        association.class,
                        association.detection_pixel.x(),
                        association.detection_pixel.y(),
                        association.field_point.x(),
                        association.field_point.y(),
                    ));
                }
            });
    }

    fn top_down_association_view(
        &mut self,
        ui: &mut Ui,
        global_debug: Option<&GlobalLocalizationDetailedDebug>,
        camera_matrix: Option<&CameraMatrix>,
        detected_objects_time: Option<SystemTime>,
    ) {
        ui.horizontal(|ui| {
            ui.heading("Top-Down Associations");
            if ui
                .button(if self.show_top_down_path {
                    "Hide path"
                } else {
                    "Show path"
                })
                .clicked()
            {
                self.show_top_down_path = !self.show_top_down_path;
            }
        });
        let available_width = ui.available_width().max(240.0);
        let (rect, response) = ui.allocate_exact_size(vec2(available_width, 220.0), Sense::hover());
        let painter = ui.painter_at(rect);
        painter.rect_filled(
            rect,
            egui::CornerRadius::same(4),
            Color32::from_rgb(24, 44, 31),
        );

        let dimensions = self.field_dimensions();
        let field_rect = top_down_field_rect(rect, &dimensions);
        draw_top_down_field_markings(&painter, field_rect, &dimensions);

        let trajectory_seconds = detected_objects_time
            .map(|time| self.recording.seconds_since_start(time))
            .unwrap_or(self.position_seconds);
        if self.show_top_down_path {
            self.draw_top_down_trajectory(&painter, field_rect, &dimensions, trajectory_seconds);
        }

        if let (Some(debug), Some(camera_matrix)) = (global_debug, camera_matrix) {
            let ground_to_field = debug.robot_to_field.inner * camera_matrix.ground_to_robot.inner;
            for association in &debug.associations {
                let ground = association.back_projected_ground;
                let detected_field =
                    ground_to_field * nalgebra::Point3::new(ground.x(), ground.y(), 0.0);
                let detected_position =
                    field_to_screen(field_rect, &dimensions, detected_field.x, detected_field.y);
                let feature_position = field_to_screen(
                    field_rect,
                    &dimensions,
                    association.field_point.x(),
                    association.field_point.y(),
                );
                let color = feature_class_color(association.class);
                painter.line_segment(
                    [detected_position, feature_position],
                    Stroke::new(1.5, color.gamma_multiply(0.75)),
                );
                painter.circle_filled(detected_position, 3.5, color);
                painter.circle_stroke(feature_position, 5.0, Stroke::new(1.5, Color32::WHITE));
            }
        }

        if let Some(debug) = global_debug {
            let robot_translation = debug.robot_to_field.inner.translation.vector;
            draw_top_down_robot_pose(
                &painter,
                field_rect,
                &dimensions,
                &debug.robot_to_field,
                Color32::from_rgb(255, 82, 190),
            );
            response.on_hover_text(format!(
                "{} associations, robot ({:.2}, {:.2})",
                debug.associations.len(),
                robot_translation.x,
                robot_translation.y
            ));
        } else {
            response.on_hover_text("no global-localization pose for this frame");
        }

        if let Some(replay_pose) = self.active_resolved_pose(trajectory_seconds) {
            let replay_pose = replay_pose.inner.cast().framed_transform();
            draw_top_down_robot_pose(
                &painter,
                field_rect,
                &dimensions,
                &replay_pose,
                Color32::from_rgb(82, 255, 140),
            );
        }
    }

    fn draw_top_down_trajectory(
        &self,
        painter: &egui::Painter,
        field_rect: Rect,
        dimensions: &FieldDimensions,
        seconds: f64,
    ) {
        if self.show_trace_history {
            for trace in &self.trace_history {
                self.draw_top_down_trace(painter, field_rect, dimensions, seconds, trace);
            }
        }

        if let Some(trajectory) = self.active_resolved_trajectory() {
            for window in trajectory
                .iter()
                .take_while(|point| point.seconds <= seconds)
                .collect::<Vec<_>>()
                .windows(2)
            {
                draw_top_down_trajectory_segment(
                    painter,
                    field_rect,
                    dimensions,
                    &window[0].robot_to_field,
                    &window[1].robot_to_field,
                );
            }
        } else {
            for window in self
                .recorded_trajectory
                .iter()
                .take_while(|point| point.seconds <= seconds)
                .collect::<Vec<_>>()
                .windows(2)
            {
                draw_top_down_trajectory_segment(
                    painter,
                    field_rect,
                    dimensions,
                    &window[0].robot_to_field,
                    &window[1].robot_to_field,
                );
            }
        }
    }

    fn draw_top_down_trace(
        &self,
        painter: &egui::Painter,
        field_rect: Rect,
        dimensions: &FieldDimensions,
        seconds: f64,
        trace: &PathTrace,
    ) {
        for window in trace
            .trajectory
            .iter()
            .take_while(|point| point.seconds <= seconds)
            .collect::<Vec<_>>()
            .windows(2)
        {
            draw_top_down_trajectory_segment_with_color(
                painter,
                field_rect,
                dimensions,
                &window[0].robot_to_field,
                &window[1].robot_to_field,
                trace.color,
            );
        }
    }

    fn timeline_panel(&mut self, context: &Context) {
        TopBottomPanel::bottom("timeline_panel").show(context, |ui| {
            ui.horizontal(|ui| {
                if ui
                    .button(if self.playing { "Pause" } else { "Play" })
                    .clicked()
                {
                    self.playing = !self.playing;
                }
                if ui.button("<").clicked() {
                    self.position_seconds = (self.position_seconds - 1.0).max(0.0);
                }
                if ui.button(">").clicked() {
                    self.position_seconds =
                        (self.position_seconds + 1.0).min(self.recording.duration().as_secs_f64());
                }
                if ui.button("< frame").clicked() {
                    self.step_frame(-1);
                }
                if ui.button("frame >").clicked() {
                    self.step_frame(1);
                }
                ui.label("speed");
                ui.add(
                    DragValue::new(&mut self.playback_rate)
                        .speed(0.1)
                        .range(0.1..=10.0),
                );
                ui.label(format!(
                    "{:.2}s / {:.2}s",
                    self.position_seconds,
                    self.recording.duration().as_secs_f64()
                ));
            });
            ui.add(
                Slider::new(
                    &mut self.position_seconds,
                    0.0..=self.recording.duration().as_secs_f64(),
                )
                .show_value(false),
            );
            if self.recording.image_count() > 0 {
                let mut frame_index = self.current_frame_index();
                let max_frame_index = self.recording.image_count() - 1;
                if ui
                    .add(Slider::new(&mut frame_index, 0..=max_frame_index).text("frame"))
                    .changed()
                {
                    self.set_frame_index(frame_index);
                }
            }
        });
    }

    fn current_frame_index(&self) -> usize {
        match &self.image_cache {
            Some(cache) => cache.image_id.index(),
            None => 0,
        }
    }

    fn step_frame(&mut self, offset: isize) {
        if self.recording.image_count() == 0 {
            return;
        }
        let max_frame_index = self.recording.image_count() - 1;
        let current = self.current_frame_index();
        let next = if offset.is_negative() {
            current.saturating_sub(offset.unsigned_abs())
        } else {
            current.saturating_add(offset as usize).min(max_frame_index)
        };
        self.set_frame_index(next);
    }

    fn set_frame_index(&mut self, frame_index: usize) {
        let Some(image_id) = self.recording.image_id_from_index(frame_index) else {
            return;
        };
        let Some(display_time) = self.recording.image_display_time(image_id) else {
            return;
        };
        self.position_seconds = self
            .recording
            .seconds_since_start(display_time)
            .clamp(0.0, self.recording.duration().as_secs_f64());
    }

    fn viewport(&mut self, context: &Context) {
        CentralPanel::default()
            .frame(egui::Frame::central_panel(&context.style()).fill(Color32::from_rgb(16, 18, 22)))
            .show(context, |ui| {
                ui.vertical(|ui| {
                    ui.horizontal(|ui| {
                        ui.heading("3D Field");
                        ui.label(RichText::new("pan/zoom with mouse").color(Color32::GRAY));
                    });
                    self.widget.ui(ui);
                });
            });
    }

    fn resolved_result(&self) -> Option<&ResolveResult> {
        match &self.resolve {
            ResolveState::Done(result) => Some(result),
            _ => None,
        }
    }
}

fn numeric_row(ui: &mut Ui, label: &str, value: &mut f64, range: std::ops::RangeInclusive<f64>) {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.add(DragValue::new(value).speed(0.1).range(range));
    });
}

fn nearest_sample(
    samples: &[crate::replay::SolveSample],
    seconds: f64,
) -> Option<&crate::replay::SolveSample> {
    if samples.is_empty() {
        return None;
    }
    let next = samples.partition_point(|sample| sample.graph_seconds <= seconds);
    nearest_by_seconds(
        next.checked_sub(1).and_then(|index| samples.get(index)),
        samples.get(next),
        seconds,
        |sample| sample.graph_seconds,
    )
    .filter(|sample| (sample.graph_seconds - seconds).abs() <= TRAJECTORY_MAX_SAMPLE_GAP_SECONDS)
}

fn nearest_trajectory_point(points: &[TrajectoryPoint], seconds: f64) -> Option<&TrajectoryPoint> {
    if points.is_empty() {
        return None;
    }
    let next = points.partition_point(|point| point.seconds <= seconds);
    nearest_by_seconds(
        next.checked_sub(1).and_then(|index| points.get(index)),
        points.get(next),
        seconds,
        |point| point.seconds,
    )
    .filter(|point| (point.seconds - seconds).abs() <= TRAJECTORY_MAX_SAMPLE_GAP_SECONDS)
}

fn nearest_by_seconds<'a, T>(
    previous: Option<&'a T>,
    next: Option<&'a T>,
    seconds: f64,
    get_seconds: impl Fn(&T) -> f64,
) -> Option<&'a T> {
    nearest_by_distance(
        previous.map(|previous| (previous, (get_seconds(previous) - seconds).abs())),
        next.map(|next| (next, (get_seconds(next) - seconds).abs())),
    )
}

fn recorded_global_debug_to_detailed(
    debug: &GlobalLocalizationDebug,
) -> GlobalLocalizationDetailedDebug {
    GlobalLocalizationDetailedDebug {
        status: recorded_global_debug_status(debug.status),
        robot_to_field: debug.robot_to_field,
        score: GlobalLocalizationScore {
            inliers: debug.inliers,
            candidate_score: debug.candidate_score,
            metric_rms_residual: debug.metric_rms_residual,
            reprojection_rmse: debug.reprojection_rmse,
            total_cost: debug.total_cost,
        },
        detections: Vec::new(),
        projected_features: Vec::new(),
        associations: Vec::new(),
    }
}

fn recorded_global_debug_status(
    status: GlobalLocalizationDebugStatus,
) -> GlobalLocalizationDetailedStatus {
    match status {
        GlobalLocalizationDebugStatus::Ambiguous => GlobalLocalizationDetailedStatus::Ambiguous,
        #[allow(deprecated)]
        GlobalLocalizationDebugStatus::Unique => GlobalLocalizationDetailedStatus::Unique,
        GlobalLocalizationDebugStatus::UniqueModuloSymmetry => {
            GlobalLocalizationDetailedStatus::UniqueModuloSymmetry
        }
    }
}

fn recorded_solve_diagnostics_summary(ui: &mut Ui, diagnostics: &SolveDiagnostics) {
    ui.label(format!(
        "recorded optimizer: {:?}",
        diagnostics.optimizer_status
    ));
    ui.label(format!(
        "recorded values/factors: {} / {}",
        diagnostics.value_count, diagnostics.factor_count
    ));
    ui.label(format!(
        "recorded total error: {:.3}",
        diagnostics.total_error
    ));
    ui.label(format!(
        "recorded VO RMS mean/max: {:.3} / {:.3}",
        diagnostics.visual_odometry.mean_rms, diagnostics.visual_odometry.max_rms
    ));
    ui.label(format!(
        "recorded visual RMS mean/max: {:.3} / {:.3}",
        diagnostics.visual_reprojection.mean_rms, diagnostics.visual_reprojection.max_rms
    ));
    ui.label(format!(
        "recorded GP RMS mean/max: {:.3} / {:.3}",
        diagnostics.gaussian_process_prior.mean_rms, diagnostics.gaussian_process_prior.max_rms
    ));
}

fn component_trace_label(parameters: &ReplayParameters, raw_vo: bool) -> String {
    if raw_vo {
        return "raw VO".to_string();
    }

    let mut parts = Vec::new();
    if parameters.include_visual_odometry {
        parts.push("VO".to_string());
    }
    if parameters.include_global_features {
        parts.push(if parameters.pose_hint.enabled {
            "visual+hint".to_string()
        } else {
            "global".to_string()
        });
    }
    if parameters.include_imu {
        let mut imu = Vec::new();
        if parameters.include_imu_roll_pitch {
            imu.push("rp");
        }
        if parameters.include_imu_yaw {
            imu.push("yaw");
        }
        if parameters.include_current_spline_orientation {
            imu.push("cur");
        }
        if parameters.include_imu_kinematics {
            imu.push("kin");
        }
        if !imu.is_empty() {
            parts.push(format!("IMU[{}]", imu.join(",")));
        }
    }
    if parameters.include_foot_heights {
        parts.push("foot".to_string());
    }

    if parts.is_empty() {
        "prior only".to_string()
    } else {
        parts.join(" + ")
    }
}

fn print_solver_samples(result: &ResolveResult) {
    println!(
        "localization_mcap_visualizer solver samples: {} samples, components: {}",
        result.samples.len(),
        component_trace_label(&result.parameters, false)
    );
    println!(
        "replay_s,graph_s,raw_x,raw_y,raw_z,raw_yaw,shown_x,shown_y,shown_z,shown_yaw,raw_dxy,raw_dyaw,shown_dxy,shown_dyaw,reused_previous,status,total_error,vo_rms,visual_rms,foot_rms,gp_rms"
    );

    let mut previous_raw = None;
    let mut previous_shown = None;
    for sample in &result.samples {
        let raw = pose_xy_z_yaw(&sample.raw_robot_to_field);
        let shown = pose_xy_z_yaw(&sample.robot_to_field);
        let raw_delta = previous_raw
            .map(|previous| pose_delta(previous, raw))
            .unwrap_or((0.0, 0.0));
        let shown_delta = previous_shown
            .map(|previous| pose_delta(previous, shown))
            .unwrap_or((0.0, 0.0));
        previous_raw = Some(raw);
        previous_shown = Some(shown);

        let diagnostics = sample.diagnostics.as_ref();
        let status = diagnostics
            .map(|diagnostics| format!("{:?}", diagnostics.optimizer_status))
            .unwrap_or_else(|| "None".to_string());
        let total_error = diagnostics.map_or(0.0, |diagnostics| diagnostics.total_error);
        let vo_rms = diagnostics.map_or(0.0, |diagnostics| diagnostics.visual_odometry.mean_rms);
        let visual_rms =
            diagnostics.map_or(0.0, |diagnostics| diagnostics.visual_reprojection.mean_rms);
        let foot_rms =
            diagnostics.map_or(0.0, |diagnostics| diagnostics.foot_above_ground.mean_rms);
        let gp_rms = diagnostics.map_or(0.0, |diagnostics| {
            diagnostics.gaussian_process_prior.mean_rms
        });

        println!(
            "{:.3},{:.3},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{},{},{:.6},{:.6},{:.6},{:.6},{:.6}",
            sample.replay_seconds,
            sample.graph_seconds,
            raw.0,
            raw.1,
            raw.2,
            raw.3,
            shown.0,
            shown.1,
            shown.2,
            shown.3,
            raw_delta.0,
            raw_delta.1,
            shown_delta.0,
            shown_delta.1,
            sample.reused_previous_pose,
            status,
            total_error,
            vo_rms,
            visual_rms,
            foot_rms,
            gp_rms,
        );
    }
}

fn pose_xy_z_yaw(pose: &linear_algebra::Isometry3<Robot, Field, f64>) -> (f64, f64, f64, f64) {
    let translation = pose.inner.translation.vector;
    let (_, _, yaw) = pose.inner.rotation.euler_angles();
    (translation.x, translation.y, translation.z, yaw)
}

fn pose_delta(previous: (f64, f64, f64, f64), current: (f64, f64, f64, f64)) -> (f64, f64) {
    let dxy = ((current.0 - previous.0).powi(2) + (current.1 - previous.1).powi(2)).sqrt();
    let dyaw = shortest_angle_distance(previous.3, current.3).abs();
    (dxy, dyaw)
}

fn shortest_angle_distance(from: f64, to: f64) -> f64 {
    let mut delta = to - from;
    while delta > std::f64::consts::PI {
        delta -= 2.0 * std::f64::consts::PI;
    }
    while delta < -std::f64::consts::PI {
        delta += 2.0 * std::f64::consts::PI;
    }
    delta
}

fn trace_color(index: usize) -> Color32 {
    const COLORS: [Color32; 10] = [
        Color32::from_rgb(255, 205, 86),
        Color32::from_rgb(54, 162, 235),
        Color32::from_rgb(255, 99, 132),
        Color32::from_rgb(75, 192, 120),
        Color32::from_rgb(200, 132, 255),
        Color32::from_rgb(255, 159, 64),
        Color32::from_rgb(96, 225, 210),
        Color32::from_rgb(230, 230, 85),
        Color32::from_rgb(170, 210, 255),
        Color32::from_rgb(255, 145, 220),
    ];
    COLORS[index % COLORS.len()]
}

fn color32_to_linear_rgba(color: Color32) -> [f32; 4] {
    [
        f32::from(color.r()) / 255.0,
        f32::from(color.g()) / 255.0,
        f32::from(color.b()) / 255.0,
        (f32::from(color.a()) / 255.0).max(0.65),
    ]
}

fn draw_detected_objects(
    ui: &mut Ui,
    image_rect: Rect,
    image_size: Vec2,
    detected_objects: &[Object<RobocupObjectLabel>],
) {
    let scale = vec2(
        image_rect.width() / image_size.x.max(1.0),
        image_rect.height() / image_size.y.max(1.0),
    );
    for object in detected_objects {
        let color = object_label_color(object.label);
        let min = image_rect.min
            + vec2(
                object.bounding_box.area.min.x() * scale.x,
                object.bounding_box.area.min.y() * scale.y,
            );
        let max = image_rect.min
            + vec2(
                object.bounding_box.area.max.x() * scale.x,
                object.bounding_box.area.max.y() * scale.y,
            );
        let rect = Rect::from_min_max(min, max).intersect(image_rect);
        let painter = ui.painter();
        painter.rect_stroke(
            rect,
            egui::CornerRadius::same(4),
            Stroke::new(2.0, color),
            StrokeKind::Outside,
        );
        let label: String = object.label.into();
        let text = format!("{label} {:.0}%", object.bounding_box.confidence * 100.0);
        let text_position = pos2(rect.min.x + 5.0, rect.min.y + 5.0);
        let galley = painter.layout_no_wrap(text, FontId::proportional(13.0), Color32::WHITE);
        let label_rect = Rect::from_min_size(
            text_position - vec2(3.0, 2.0),
            galley.size() + vec2(6.0, 4.0),
        );
        painter.rect_filled(
            label_rect,
            egui::CornerRadius::same(3),
            color.gamma_multiply(0.85),
        );
        painter.galley(text_position, galley, Color32::WHITE);
    }
}

fn draw_recorded_association_pixels(
    ui: &mut Ui,
    image_rect: Rect,
    image_size: Vec2,
    field_mark_associations: &FieldMarkAssociations,
) {
    let scale = vec2(
        image_rect.width() / image_size.x.max(1.0),
        image_rect.height() / image_size.y.max(1.0),
    );
    let painter = ui.painter();
    for association in &field_mark_associations.associations {
        let position = image_rect.min
            + vec2(
                association.detection.x() * scale.x,
                association.detection.y() * scale.y,
            );
        if !image_rect.contains(position) {
            continue;
        }
        let color = match association.kind {
            FieldMarkAssociationKind::GlobalUnique => Color32::from_rgb(120, 255, 170),
            FieldMarkAssociationKind::PoseHint => Color32::from_rgb(120, 180, 255),
        };
        painter.circle_stroke(position, 7.0, Stroke::new(2.0, color));
        painter.line_segment(
            [position - vec2(5.0, 0.0), position + vec2(5.0, 0.0)],
            Stroke::new(1.5, color),
        );
        painter.line_segment(
            [position - vec2(0.0, 5.0), position + vec2(0.0, 5.0)],
            Stroke::new(1.5, color),
        );
    }
}

fn image_hover_info(
    response: &egui::Response,
    image_size: Vec2,
    detected_objects: &[Object<RobocupObjectLabel>],
    global_debug: Option<&GlobalLocalizationDetailedDebug>,
) -> Vec<String> {
    let Some(pointer) = response.hover_pos() else {
        return Vec::new();
    };
    let Some(pixel) = screen_to_image_pixel(response.rect, image_size, pointer) else {
        return Vec::new();
    };
    let mut lines = Vec::new();

    for object in detected_objects {
        let min = vec2(
            object.bounding_box.area.min.x(),
            object.bounding_box.area.min.y(),
        );
        let max = vec2(
            object.bounding_box.area.max.x(),
            object.bounding_box.area.max.y(),
        );
        if pixel.x >= min.x && pixel.x <= max.x && pixel.y >= min.y && pixel.y <= max.y {
            let label: String = object.label.into();
            lines.push(format!(
                "detection: {label} {:.0}% bbox ({:.0},{:.0})-({:.0},{:.0})",
                object.bounding_box.confidence * 100.0,
                min.x,
                min.y,
                max.x,
                max.y,
            ));
        }
    }

    if let Some(debug) = global_debug {
        for detection in &debug.detections {
            if pixel_distance(pixel, vec2(detection.pixel.x(), detection.pixel.y())) <= 10.0 {
                lines.push(format!(
                    "feature detection #{} {:?} pixel ({:.1},{:.1}) ground ({:.2},{:.2})",
                    detection.index,
                    detection.class,
                    detection.pixel.x(),
                    detection.pixel.y(),
                    detection.ground.x(),
                    detection.ground.y(),
                ));
            }
        }
        for projection in &debug.projected_features {
            let Some(projected_pixel) = projection.projected_pixel else {
                continue;
            };
            if pixel_distance(pixel, vec2(projected_pixel.x(), projected_pixel.y())) <= 10.0 {
                lines.push(format!(
                    "projection #{} sym #{} {:?} field ({:.2},{:.2}) {}",
                    projection.index,
                    projection.symmetric_index,
                    projection.class,
                    projection.field_point.x(),
                    projection.field_point.y(),
                    if projection.accepted {
                        "accepted"
                    } else {
                        "candidate"
                    },
                ));
            }
        }
        for association in &debug.associations {
            let detection_distance = pixel_distance(
                pixel,
                vec2(
                    association.detection_pixel.x(),
                    association.detection_pixel.y(),
                ),
            );
            let projection_distance = match association.projected_pixel {
                Some(projected_pixel) => {
                    pixel_distance(pixel, vec2(projected_pixel.x(), projected_pixel.y()))
                }
                None => f32::INFINITY,
            };
            if detection_distance <= 10.0 || projection_distance <= 10.0 {
                let error = match association.reprojection_error_px {
                    Some(error) => format!("{error:.1}px"),
                    None => "not visible".to_string(),
                };
                lines.push(format!(
                    "association det #{} -> feature #{} {:?} error {error} field ({:.2},{:.2})",
                    association.detection_index,
                    association.feature_index,
                    association.class,
                    association.field_point.x(),
                    association.field_point.y(),
                ));
            }
        }
    }

    lines
}

fn screen_to_image_pixel(image_rect: Rect, image_size: Vec2, pointer: egui::Pos2) -> Option<Vec2> {
    if !image_rect.contains(pointer) {
        return None;
    }
    Some(vec2(
        (pointer.x - image_rect.left()) / image_rect.width().max(1.0) * image_size.x,
        (pointer.y - image_rect.top()) / image_rect.height().max(1.0) * image_size.y,
    ))
}

fn pixel_distance(left: Vec2, right: Vec2) -> f32 {
    (left - right).length()
}

fn draw_global_debug_overlay(
    ui: &mut Ui,
    image_rect: Rect,
    image_size: Vec2,
    debug: &GlobalLocalizationDetailedDebug,
) {
    let scale = vec2(
        image_rect.width() / image_size.x.max(1.0),
        image_rect.height() / image_size.y.max(1.0),
    );
    let painter = ui.painter();

    for projection in &debug.projected_features {
        let Some(pixel) = projection.projected_pixel else {
            continue;
        };
        let position = image_rect.min + vec2(pixel.x() * scale.x, pixel.y() * scale.y);
        if !image_rect.contains(position) {
            continue;
        }
        let color = if projection.accepted {
            Color32::WHITE
        } else {
            feature_class_color(projection.class).gamma_multiply(0.45)
        };
        painter.circle_stroke(position, 4.0, Stroke::new(1.5, color));
    }

    for association in &debug.associations {
        let Some(projected) = association.projected_pixel else {
            continue;
        };
        let detection = image_rect.min
            + vec2(
                association.detection_pixel.x() * scale.x,
                association.detection_pixel.y() * scale.y,
            );
        let projection = image_rect.min + vec2(projected.x() * scale.x, projected.y() * scale.y);
        let color = feature_class_color(association.class);
        painter.line_segment([detection, projection], Stroke::new(2.0, color));
        painter.circle_filled(detection, 4.0, color);
        painter.circle_filled(projection, 3.0, Color32::WHITE);
    }
}

fn top_down_field_rect(rect: Rect, dimensions: &FieldDimensions) -> Rect {
    let field_length = dimensions.length.max(1.0);
    let field_width = dimensions.width.max(1.0);
    let scale = (rect.width() / field_length).min(rect.height() / field_width) * 0.92;
    Rect::from_center_size(
        rect.center(),
        vec2(field_length * scale, field_width * scale),
    )
}

fn draw_top_down_field_markings(
    painter: &egui::Painter,
    field_rect: Rect,
    dimensions: &FieldDimensions,
) {
    let white = Color32::from_rgb(235, 245, 238);
    let muted = Color32::from_rgb(150, 168, 156);
    let line = Stroke::new(1.5, white);
    let thin_line = Stroke::new(1.0, muted);

    painter.rect_stroke(
        field_rect,
        egui::CornerRadius::same(2),
        line,
        StrokeKind::Inside,
    );
    draw_top_down_field_segment(
        painter,
        field_rect,
        dimensions,
        0.0,
        -dimensions.width / 2.0,
        0.0,
        dimensions.width / 2.0,
        thin_line,
    );

    let center = field_to_screen(field_rect, dimensions, 0.0, 0.0);
    let center_circle_radius =
        dimensions.center_circle_diameter / 2.0 / dimensions.length.max(1.0) * field_rect.width();
    painter.circle_stroke(center, center_circle_radius, thin_line);

    for side in [-1.0, 1.0] {
        draw_top_down_goal_area(
            painter,
            field_rect,
            dimensions,
            side,
            dimensions.goal_box_area_length,
            dimensions.goal_box_area_width,
            thin_line,
        );
        draw_top_down_goal_area(
            painter,
            field_rect,
            dimensions,
            side,
            dimensions.penalty_area_length,
            dimensions.penalty_area_width,
            thin_line,
        );
        let penalty_x = side * (dimensions.length / 2.0 - dimensions.penalty_marker_distance);
        painter.circle_filled(
            field_to_screen(field_rect, dimensions, penalty_x, 0.0),
            2.0,
            white,
        );
    }
}

fn draw_top_down_goal_area(
    painter: &egui::Painter,
    field_rect: Rect,
    dimensions: &FieldDimensions,
    side: f32,
    length: f32,
    width: f32,
    stroke: Stroke,
) {
    let goal_line_x = side * dimensions.length / 2.0;
    let inner_x = goal_line_x - side * length;
    let half_width = width / 2.0;
    draw_top_down_field_segment(
        painter,
        field_rect,
        dimensions,
        goal_line_x,
        -half_width,
        inner_x,
        -half_width,
        stroke,
    );
    draw_top_down_field_segment(
        painter,
        field_rect,
        dimensions,
        inner_x,
        -half_width,
        inner_x,
        half_width,
        stroke,
    );
    draw_top_down_field_segment(
        painter,
        field_rect,
        dimensions,
        inner_x,
        half_width,
        goal_line_x,
        half_width,
        stroke,
    );
}

#[allow(clippy::too_many_arguments)]
fn draw_top_down_field_segment(
    painter: &egui::Painter,
    field_rect: Rect,
    dimensions: &FieldDimensions,
    start_x: f32,
    start_y: f32,
    end_x: f32,
    end_y: f32,
    stroke: Stroke,
) {
    painter.line_segment(
        [
            field_to_screen(field_rect, dimensions, start_x, start_y),
            field_to_screen(field_rect, dimensions, end_x, end_y),
        ],
        stroke,
    );
}

fn field_to_screen(
    field_rect: Rect,
    dimensions: &FieldDimensions,
    field_x: f32,
    field_y: f32,
) -> egui::Pos2 {
    let x = field_rect.center().x + field_x / dimensions.length.max(1.0) * field_rect.width();
    let y = field_rect.center().y - field_y / dimensions.width.max(1.0) * field_rect.height();
    pos2(x, y)
}

fn draw_top_down_trajectory_segment(
    painter: &egui::Painter,
    field_rect: Rect,
    dimensions: &FieldDimensions,
    start: &linear_algebra::Isometry3<Robot, Field, f64>,
    end: &linear_algebra::Isometry3<Robot, Field, f64>,
) {
    draw_top_down_trajectory_segment_with_color(
        painter,
        field_rect,
        dimensions,
        start,
        end,
        Color32::from_rgb(82, 170, 255).gamma_multiply(0.55),
    );
}

fn draw_top_down_trajectory_segment_with_color(
    painter: &egui::Painter,
    field_rect: Rect,
    dimensions: &FieldDimensions,
    start: &linear_algebra::Isometry3<Robot, Field, f64>,
    end: &linear_algebra::Isometry3<Robot, Field, f64>,
    color: Color32,
) {
    let start = start.inner.translation.vector;
    let end = end.inner.translation.vector;
    if !start.x.is_finite() || !start.y.is_finite() || !end.x.is_finite() || !end.y.is_finite() {
        return;
    }
    painter.line_segment(
        [
            field_to_screen(field_rect, dimensions, start.x as f32, start.y as f32),
            field_to_screen(field_rect, dimensions, end.x as f32, end.y as f32),
        ],
        Stroke::new(1.5, color),
    );
}

fn draw_top_down_robot_pose(
    painter: &egui::Painter,
    field_rect: Rect,
    dimensions: &FieldDimensions,
    robot_to_field: &linear_algebra::Isometry3<Robot, Field>,
    color: Color32,
) {
    let robot_translation = robot_to_field.inner.translation.vector;
    if !robot_translation.x.is_finite() || !robot_translation.y.is_finite() {
        return;
    }

    let center = field_to_screen(
        field_rect,
        dimensions,
        robot_translation.x,
        robot_translation.y,
    );
    let forward_in_field = robot_to_field
        .inner
        .transform_point(&nalgebra::Point3::new(0.45, 0.0, 0.0));
    let tip = field_to_screen(
        field_rect,
        dimensions,
        forward_in_field.x,
        forward_in_field.y,
    );
    let arrow = tip - center;

    painter.circle_filled(center, 4.0, color);
    if arrow.length_sq() <= 1.0 {
        return;
    }

    let direction = arrow.normalized();
    let perpendicular = vec2(-direction.y, direction.x);
    painter.line_segment([center, tip], Stroke::new(2.0, color));
    painter.add(egui::Shape::convex_polygon(
        vec![
            tip,
            tip - direction * 8.0 + perpendicular * 4.0,
            tip - direction * 8.0 - perpendicular * 4.0,
        ],
        color,
        Stroke::NONE,
    ));
}

fn object_label_color(label: RobocupObjectLabel) -> Color32 {
    match label {
        RobocupObjectLabel::Ball => Color32::from_rgb(255, 145, 64),
        RobocupObjectLabel::GoalPost => Color32::from_rgb(245, 245, 245),
        RobocupObjectLabel::Robot => Color32::from_rgb(82, 170, 255),
        RobocupObjectLabel::PenaltySpot => Color32::from_rgb(255, 230, 96),
        RobocupObjectLabel::LSpot | RobocupObjectLabel::TSpot | RobocupObjectLabel::XSpot => {
            Color32::from_rgb(120, 255, 170)
        }
    }
}

fn feature_class_color(class: VisualFeatureClass) -> Color32 {
    match class {
        VisualFeatureClass::GoalPost => Color32::WHITE,
        VisualFeatureClass::LSpot => Color32::from_rgb(120, 255, 170),
        VisualFeatureClass::TSpot => Color32::from_rgb(120, 180, 255),
        VisualFeatureClass::PenaltySpot => Color32::from_rgb(255, 230, 96),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StereoSide {
    Left,
    Right,
}

#[derive(Clone)]
struct CachedStereoFrame {
    image_id: StereoImageId,
    frame: StereoFrame,
}

impl CachedStereoFrame {
    fn active(&self, side: StereoSide) -> &CameraImage {
        match side {
            StereoSide::Left => &self.frame.left,
            StereoSide::Right => &self.frame.right,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CameraMatrixKey {
    time_nanos: i64,
    calibrated_intrinsics_time_nanos: Option<u128>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct GlobalDebugKey {
    camera_matrix_key: Option<CameraMatrixKey>,
    detected_objects_time_nanos: Option<u128>,
    recorded_debug_time_nanos: Option<u128>,
    pose_revision: SceneVersion,
    global_localizer: GlobalLocalizerParameters,
}

#[derive(Default)]
struct CachedGlobalDebug {
    key: Option<GlobalDebugKey>,
    version: SceneVersion,
    debug: Option<Arc<GlobalLocalizationDetailedDebug>>,
}

enum ResolveState {
    Idle,
    Running {
        receiver: Receiver<ResolveMessage>,
        cancel: Arc<AtomicBool>,
        progress: ResolveProgress,
    },
    Done(ResolveResult),
    Failed(String),
}
