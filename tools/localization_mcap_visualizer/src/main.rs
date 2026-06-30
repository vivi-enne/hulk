use std::{
    path::PathBuf,
    sync::Arc,
    time::{Duration, SystemTime},
};

use clap::Parser;
use color_eyre::{Result, eyre::eyre};
use eframe::{
    NativeOptions, Renderer,
    egui_wgpu::{WgpuConfiguration, WgpuSetup},
    run_native,
};
use stereo_visual_odometry::{
    VisualOdometryPipeline, parameters::StereoVisualOdometryPoseEstimationParameters,
};
use tracing_subscriber::EnvFilter;

use crate::app::LocalizationMcapVisualizerApp;
use crate::mcap_recording::{EventKind, RecordedEvent, Recording, TrajectoryPoint, nanos_abs_diff};
use crate::replay::{
    ReplayParameters, ReplayVisualOdometryEvent, TimestampMode, resolve_recording,
    resolve_recording_with_visual_odometry_override,
};

mod app;
mod mcap_recording;
mod replay;
mod scene;

#[derive(Clone, Debug, Parser)]
struct Arguments {
    /// Localization recorder MCAP file to inspect.
    #[arg(default_value = "recording.mcap")]
    mcap: PathBuf,
    /// Run replay headlessly and print trajectory stability metrics.
    #[arg(long)]
    resolve_summary: bool,
    /// Run headless replay with only IMU orientation factors and priors enabled.
    #[arg(long)]
    orientation_only_summary: bool,
    /// Recompute field-feature associations from detections during headless replay.
    #[arg(long)]
    recompute_global_features: bool,
    /// Override visual-odometry covariance during headless replay.
    #[arg(long)]
    vo_covariance: Option<f64>,
    /// Seconds to ignore at the start of resolve-summary trajectory metrics.
    #[arg(long, default_value_t = 5.0)]
    metrics_skip_seconds: f64,
    /// Drop compensated visual-odometry translations above this threshold in meters.
    #[arg(long)]
    max_vo_translation: Option<f32>,
    /// Drop compensated visual-odometry rotations above this threshold in degrees.
    #[arg(long)]
    max_vo_rotation_deg: Option<f32>,
    /// Disable visual odometry during headless replay.
    #[arg(long)]
    no_visual_odometry: bool,
    /// Disable field-feature associations during headless replay.
    #[arg(long)]
    no_global_features: bool,
    /// Disable IMU orientation factors during headless replay.
    #[arg(long)]
    no_imu: bool,
    /// Disable foot-height factors during headless replay.
    #[arg(long)]
    no_foot_heights: bool,
    /// Override pose-hint visual-feature covariance during headless replay.
    #[arg(long)]
    pose_hint_noise: Option<f64>,
    /// Override global visual-feature covariance during headless replay.
    #[arg(long)]
    visual_noise: Option<f64>,
    /// Override minimum pose-hint features per ingested frame during headless replay.
    #[arg(long)]
    pose_hint_min_features: Option<usize>,
    /// Override weak pose-hint frames required before global recovery is accepted.
    #[arg(long)]
    recovery_frames: Option<usize>,
    /// Override maximum global-recovery translation disagreement in meters.
    #[arg(long)]
    recovery_max_distance: Option<f32>,
    /// Override maximum global-recovery yaw disagreement in degrees.
    #[arg(long)]
    recovery_max_angle_deg: Option<f32>,
    /// Override global localization metric RMS threshold during headless replay.
    #[arg(long)]
    global_rms_threshold: Option<f32>,
    /// Override global localization minimum inliers during headless replay.
    #[arg(long)]
    global_min_inliers: Option<usize>,
    /// Override pose-hint healthy RMSE threshold during headless replay.
    #[arg(long)]
    pose_hint_healthy_rmse: Option<f32>,
    /// Override pose-hint second-best projection margin during headless replay.
    #[arg(long)]
    pose_hint_margin: Option<f32>,
    /// Print visual-odometry compensation diagnostics without opening the UI.
    #[arg(long)]
    vo_diagnostics: bool,
    /// Regenerate visual odometry from recorded stereo frames and print diagnostics.
    #[arg(long)]
    regenerate_vo_diagnostics: bool,
    /// Replay localization using regenerated visual odometry instead of recorded visual odometry.
    #[arg(long)]
    resolve_regenerated_vo_summary: bool,
    /// Stereo baseline in meters used when reconstructing missing stereo_camera_info.
    #[arg(long, default_value_t = 0.05)]
    stereo_baseline_m: f64,
    /// XFeat/LightGlue ONNX model path for regenerated visual odometry.
    #[arg(long, default_value = "etc/neural_networks/xfeat-lighterglue.onnx")]
    vo_model: PathBuf,
}

fn main() -> Result<()> {
    color_eyre::install()?;
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let arguments = Arguments::parse();
    let recording = Arc::new(mcap_recording::Recording::load(&arguments.mcap)?);

    if arguments.orientation_only_summary {
        print_orientation_only_summary(&recording)?;
        return Ok(());
    }
    if arguments.resolve_summary {
        print_resolve_summary(
            &recording,
            replay_parameters(&arguments),
            arguments.metrics_skip_seconds,
        )?;
        return Ok(());
    }
    if arguments.vo_diagnostics {
        print_vo_diagnostics(&recording)?;
        return Ok(());
    }
    if arguments.regenerate_vo_diagnostics {
        print_regenerated_vo_diagnostics(
            &recording,
            &arguments.vo_model,
            arguments.stereo_baseline_m,
        )?;
        return Ok(());
    }
    if arguments.resolve_regenerated_vo_summary {
        print_resolve_regenerated_vo_summary(
            &recording,
            &arguments.vo_model,
            arguments.stereo_baseline_m,
        )?;
        return Ok(());
    }

    run_native(
        "Localization MCAP Visualizer",
        NativeOptions {
            renderer: Renderer::Wgpu,
            wgpu_options: wgpu_options(),
            ..Default::default()
        },
        Box::new(move |creation_context| {
            let app = LocalizationMcapVisualizerApp::new(
                creation_context,
                arguments.mcap.clone(),
                recording.clone(),
            )?;
            Ok(Box::new(app))
        }),
    )
    .map_err(|error| eyre!("failed to run localization MCAP visualizer: {error}"))?;

    Ok(())
}

fn replay_parameters(arguments: &Arguments) -> ReplayParameters {
    let mut parameters = ReplayParameters {
        recompute_global_features: arguments.recompute_global_features,
        ..ReplayParameters::default()
    };
    if let Some(vo_covariance) = arguments.vo_covariance {
        parameters.override_visual_odometry_covariance = true;
        parameters.visual_odometry_covariance = vo_covariance;
    }
    parameters.max_visual_odometry_translation = arguments.max_vo_translation;
    parameters.max_visual_odometry_rotation = arguments.max_vo_rotation_deg.map(f32::to_radians);
    if arguments.no_visual_odometry {
        parameters.include_visual_odometry = false;
    }
    if arguments.no_global_features {
        parameters.include_global_association = false;
        parameters.include_pose_hint_association = false;
    }
    if arguments.no_imu {
        parameters.include_imu = false;
    }
    if arguments.no_foot_heights {
        parameters.include_foot_heights = false;
    }
    if let Some(pose_hint_noise) = arguments.pose_hint_noise {
        parameters.pose_hint_visual_feature_noise_variance = pose_hint_noise;
    }
    if let Some(visual_noise) = arguments.visual_noise {
        parameters.visual_feature_noise_variance = visual_noise;
    }
    if let Some(pose_hint_min_features) = arguments.pose_hint_min_features {
        parameters.pose_hint_visual_min_features_per_frame = pose_hint_min_features;
    }
    if let Some(recovery_frames) = arguments.recovery_frames {
        parameters.pose_hint.recovery_frames = recovery_frames.max(1);
    }
    if let Some(recovery_max_distance) = arguments.recovery_max_distance {
        parameters.pose_hint.recovery_max_pose_distance = recovery_max_distance;
    }
    if let Some(recovery_max_angle_deg) = arguments.recovery_max_angle_deg {
        parameters.pose_hint.recovery_max_pose_angle = recovery_max_angle_deg.to_radians();
    }
    if let Some(global_rms_threshold) = arguments.global_rms_threshold {
        parameters.global_localizer.rms_threshold = global_rms_threshold;
    }
    if let Some(global_min_inliers) = arguments.global_min_inliers {
        parameters.global_localizer.min_inliers = global_min_inliers;
    }
    if let Some(pose_hint_healthy_rmse) = arguments.pose_hint_healthy_rmse {
        parameters.pose_hint.healthy_max_rmse_px = pose_hint_healthy_rmse;
    }
    if let Some(pose_hint_margin) = arguments.pose_hint_margin {
        parameters.pose_hint.second_best_reprojection_margin_px = pose_hint_margin;
    }
    parameters
}

fn print_regenerated_vo_diagnostics(
    recording: &Recording,
    model_path: &PathBuf,
    stereo_baseline_m: f64,
) -> Result<()> {
    let regenerated = regenerate_visual_odometry(recording, model_path, stereo_baseline_m)?;
    print_regenerated_vo_summary(&regenerated, stereo_baseline_m);
    Ok(())
}

fn print_resolve_regenerated_vo_summary(
    recording: &Recording,
    model_path: &PathBuf,
    stereo_baseline_m: f64,
) -> Result<()> {
    let regenerated = regenerate_visual_odometry(recording, model_path, stereo_baseline_m)?;
    print_regenerated_vo_summary(&regenerated, stereo_baseline_m);
    let replay = resolve_recording_with_visual_odometry_override(
        recording,
        ReplayParameters::default(),
        &regenerated.events,
    )?;
    println!(
        "recorded_localization {}",
        trajectory_metrics(&recording.recorded_localization_trajectory())
    );
    println!(
        "regenerated_vo_replayed_localization {}",
        trajectory_metrics(&replay.trajectory())
    );
    println!(
        "replay_stats imu_ingested={} foot_heights_ingested={} vo_received={} vo_ingested={} vo_dropped_invalid={} vo_dropped_gated={} vo_stale_camera_skips={} global_candidates={} global_frames_ingested={} global_associations_ingested={} global_unique_frames_ingested={} global_unique_associations_ingested={} pose_hint_frames_ingested={} pose_hint_associations_ingested={} solve_samples={}",
        replay.stats.imu_ingested,
        replay.stats.foot_heights_ingested,
        replay.stats.vo_received,
        replay.stats.vo_ingested,
        replay.stats.vo_dropped_invalid,
        replay.stats.vo_dropped_gated,
        replay.stats.vo_skipped_stale_camera_matrix,
        replay.stats.global_candidates,
        replay.stats.global_frames_ingested,
        replay.stats.global_associations_ingested,
        replay.stats.global_unique_frames_ingested,
        replay.stats.global_unique_associations_ingested,
        replay.stats.pose_hint_frames_ingested,
        replay.stats.pose_hint_associations_ingested,
        replay.samples.len(),
    );
    Ok(())
}

struct RegeneratedVisualOdometry {
    events: Vec<ReplayVisualOdometryEvent>,
    stats: VoDiagnosticStats,
    frames: usize,
    estimates: usize,
    failed: usize,
}

fn regenerate_visual_odometry(
    recording: &Recording,
    model_path: &PathBuf,
    stereo_baseline_m: f64,
) -> Result<RegeneratedVisualOdometry> {
    if recording.image_count() == 0 {
        return Err(eyre!(
            "recording contains no raw stereo frames; enable localization_recorder.include_raw_images"
        ));
    }
    if !stereo_baseline_m.is_finite() || stereo_baseline_m <= 0.0 {
        return Err(eyre!("--stereo-baseline-m must be finite and > 0"));
    }

    let stereo_camera_info = if let Some(stereo_camera_info) = &recording.stereo_camera_info {
        stereo_camera_info.clone()
    } else {
        let first_image_id = recording
            .image_id_from_index(0)
            .ok_or_else(|| eyre!("recording contains no raw stereo frames"))?;
        let first_stereo_pair = recording.decode_stereo_pair(first_image_id)?;
        stereo_camera_info_from_recording(
            &recording.first_camera_matrix,
            &first_stereo_pair.inner,
            stereo_baseline_m,
        )
    };
    let parameters = default_vo_pose_estimation_parameters();
    let mut pipeline = VisualOdometryPipeline::new(model_path, stereo_camera_info)?;
    let camera_matrices = recorded_camera_matrices(recording);
    let mut stats = VoDiagnosticStats::default();
    let mut frames = 0usize;
    let mut odometry_estimates = 0usize;
    let mut failed_odometry = 0usize;
    let mut previous_time: Option<ros_z::time::Time> = None;
    let mut events = Vec::new();

    for index in 0..recording.image_count() {
        let Some(image_id) = recording.image_id_from_index(index) else {
            continue;
        };
        let stereo_pair = recording.decode_stereo_pair(image_id)?;
        frames += 1;
        let current_time = stereo_pair.time;
        let had_previous = previous_time.is_some();
        let odometry = pipeline.process(&stereo_pair.inner, &parameters)?;
        if had_previous && odometry.is_none() {
            failed_odometry += 1;
            pipeline.reset_tracking();
            previous_time = None;
            continue;
        }
        if let (Some(previous_time), Some(previous_left_camera_to_current_left_camera)) =
            (previous_time, odometry)
        {
            let measured_current_camera_to_previous_camera =
                previous_left_camera_to_current_left_camera.inverse();
            let previous_wallclock = previous_time.to_wallclock();
            let current_wallclock = current_time.to_wallclock();
            let Some(previous_camera_matrix) = nearest_camera_matrix(
                &camera_matrices,
                previous_wallclock,
                TimestampMode::Embedded,
            ) else {
                stats.missing_camera_matrix += 1;
                continue;
            };
            let Some(current_camera_matrix) =
                nearest_camera_matrix(&camera_matrices, current_wallclock, TimestampMode::Embedded)
            else {
                stats.missing_camera_matrix += 1;
                continue;
            };
            stats.add_sample(
                previous_camera_matrix.distance,
                current_camera_matrix.distance,
                measured_current_camera_to_previous_camera,
                &previous_camera_matrix.matrix.matrix.inner,
                &current_camera_matrix.matrix.matrix.inner,
            );
            let publish_time = recording
                .aligned_image_time(current_time)
                .unwrap_or(current_wallclock);
            let log_time = recording.image_log_time(image_id).unwrap_or(publish_time);
            events.push(ReplayVisualOdometryEvent {
                log_time,
                publish_time,
                delta: types::visual_odometry::VisualOdometryDelta {
                    previous_time,
                    current_time,
                    current_left_camera_to_previous_left_camera:
                        measured_current_camera_to_previous_camera,
                },
            });
            odometry_estimates += 1;
        }
        previous_time = Some(current_time);
    }

    Ok(RegeneratedVisualOdometry {
        events,
        stats,
        frames,
        estimates: odometry_estimates,
        failed: failed_odometry,
    })
}

fn print_regenerated_vo_summary(regenerated: &RegeneratedVisualOdometry, stereo_baseline_m: f64) {
    let mut stats = regenerated.stats.clone();
    println!(
        "regenerated_vo frames={} estimates={} failed={} baseline_m={:.4} {}",
        regenerated.frames,
        regenerated.estimates,
        regenerated.failed,
        stereo_baseline_m,
        stats.summary()
    );
}

fn stereo_camera_info_from_recording(
    camera_matrix: &projection::camera_matrix::CameraMatrix,
    stereo_pair: &types::stereo_image_pair::StereoImagePair,
    baseline_m: f64,
) -> types::stereo_camera_info::StereoCameraInfo {
    let width = stereo_pair.left.width;
    let height = stereo_pair.left.height;
    let fx = camera_matrix.intrinsics.focals.x as f64;
    let fy = camera_matrix.intrinsics.focals.y as f64;
    let cx = camera_matrix.intrinsics.optical_center.x() as f64;
    let cy = camera_matrix.intrinsics.optical_center.y() as f64;

    types::stereo_camera_info::StereoCameraInfo {
        left: reconstructed_camera_info(width, height, fx, fy, cx, cy, 0.0),
        right: reconstructed_camera_info(width, height, fx, fy, cx, cy, -fx * baseline_m),
    }
}

fn reconstructed_camera_info(
    width: u32,
    height: u32,
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    tx: f64,
) -> ros2::sensor_msgs::camera_info::CameraInfo {
    ros2::sensor_msgs::camera_info::CameraInfo {
        width,
        height,
        k: [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
        r: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        p: [fx, 0.0, cx, tx, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0],
        ..Default::default()
    }
}

fn default_vo_pose_estimation_parameters() -> StereoVisualOdometryPoseEstimationParameters {
    StereoVisualOdometryPoseEstimationParameters {
        minimum_pnp_correspondences: 8,
        ransac_reprojection_threshold_px: 6.0,
        ransac_max_iterations: 100,
        ransac_confidence: 0.99,
        lm_max_iterations: 20,
        lm_initial_lambda: 0.001,
        lm_min_lambda: 0.0000001,
        lm_max_lambda: 10000000000.0,
        lm_step_tolerance: 0.000001,
        lm_cost_tolerance: 0.000001,
        lm_huber_threshold_px: 3.0,
        full_weight_disparity_px: 8.0,
        min_disparity_weight: 0.5,
        max_vertical_disparity_px: 3.0,
    }
}

fn print_vo_diagnostics(recording: &Recording) -> Result<()> {
    let camera_matrices = recorded_camera_matrices(recording);
    for mode in [TimestampMode::Embedded, TimestampMode::McapPublish] {
        let mut previous_mcap_publish_time = None;
        let mut stats = VoDiagnosticStats::default();
        for event in recording.events() {
            let EventKind::VisualOdometry(delta) = &event.kind else {
                continue;
            };
            let Some((previous_time, current_time)) = diagnostic_measurement_times(
                recording,
                event,
                delta,
                mode,
                &mut previous_mcap_publish_time,
            ) else {
                continue;
            };
            let Some(previous_camera_matrix) =
                nearest_camera_matrix(&camera_matrices, previous_time, mode)
            else {
                stats.missing_camera_matrix += 1;
                continue;
            };
            let Some(current_camera_matrix) =
                nearest_camera_matrix(&camera_matrices, current_time, mode)
            else {
                stats.missing_camera_matrix += 1;
                continue;
            };
            stats.add_sample(
                previous_camera_matrix.distance,
                current_camera_matrix.distance,
                delta.current_left_camera_to_previous_left_camera,
                &previous_camera_matrix.matrix.matrix.inner,
                &current_camera_matrix.matrix.matrix.inner,
            );
        }

        println!("vo_diagnostics mode={mode:?} {}", stats.summary());
    }
    Ok(())
}

fn diagnostic_measurement_times(
    recording: &Recording,
    event: &RecordedEvent,
    delta: &types::visual_odometry::VisualOdometryDelta,
    mode: TimestampMode,
    previous_mcap_publish_time: &mut Option<SystemTime>,
) -> Option<(SystemTime, SystemTime)> {
    match mode {
        TimestampMode::Embedded => Some((
            delta.previous_time.to_wallclock(),
            delta.current_time.to_wallclock(),
        )),
        TimestampMode::McapPublish => {
            let current = recording
                .aligned_image_time(delta.current_time)
                .unwrap_or(event.publish_time);
            let previous = recording
                .aligned_image_time(delta.previous_time)
                .or(*previous_mcap_publish_time)
                .or_else(|| {
                    let embedded_duration = delta
                        .current_time
                        .to_wallclock()
                        .duration_since(delta.previous_time.to_wallclock())
                        .ok()?;
                    current.checked_sub(embedded_duration)
                });
            *previous_mcap_publish_time = Some(current);
            previous.map(|previous| (previous, current))
        }
    }
}

fn recorded_camera_matrices(recording: &Recording) -> Vec<DiagnosticCameraMatrix<'_>> {
    recording
        .events()
        .iter()
        .filter_map(|event| match &event.kind {
            EventKind::CameraMatrix(matrix) => Some(DiagnosticCameraMatrix {
                publish_time: event.publish_time,
                matrix,
            }),
            _ => None,
        })
        .collect()
}

fn nearest_camera_matrix<'a>(
    camera_matrices: &'a [DiagnosticCameraMatrix<'a>],
    time: SystemTime,
    mode: TimestampMode,
) -> Option<NearestDiagnosticCameraMatrix<'a>> {
    camera_matrices
        .iter()
        .min_by_key(|candidate| nanos_abs_diff(candidate.time(mode), time))
        .map(|matrix| NearestDiagnosticCameraMatrix {
            matrix,
            distance: Duration::from_nanos(
                nanos_abs_diff(matrix.time(mode), time).min(u64::MAX as u128) as u64,
            ),
        })
}

struct DiagnosticCameraMatrix<'a> {
    publish_time: SystemTime,
    matrix: &'a types::time_wrapper::TimeWrapper<projection::camera_matrix::CameraMatrix>,
}

impl DiagnosticCameraMatrix<'_> {
    fn time(&self, mode: TimestampMode) -> SystemTime {
        match mode {
            TimestampMode::Embedded => self.matrix.time.to_wallclock(),
            TimestampMode::McapPublish => self.publish_time,
        }
    }
}

struct NearestDiagnosticCameraMatrix<'a> {
    matrix: &'a DiagnosticCameraMatrix<'a>,
    distance: Duration,
}

#[derive(Clone, Default)]
struct VoDiagnosticStats {
    samples: usize,
    missing_camera_matrix: usize,
    previous_camera_age_ms: Vec<f64>,
    current_camera_age_ms: Vec<f64>,
    measured_translation_m: Vec<f64>,
    measured_rotation_deg: Vec<f64>,
    expected_translation_m: Vec<f64>,
    expected_rotation_deg: Vec<f64>,
    compensated_translation_m: Vec<f64>,
    compensated_rotation_deg: Vec<f64>,
    inverted_compensated_translation_m: Vec<f64>,
    inverted_compensated_rotation_deg: Vec<f64>,
    measured_expected_error_translation_m: Vec<f64>,
    measured_expected_error_rotation_deg: Vec<f64>,
    measured_inverse_expected_error_translation_m: Vec<f64>,
    measured_inverse_expected_error_rotation_deg: Vec<f64>,
    measured_expected_translation_cosine: Vec<f64>,
    measured_expected_translation_ratio: Vec<f64>,
    measured_expected_rotation_cosine: Vec<f64>,
    measured_expected_rotation_ratio: Vec<f64>,
    compensated_translation_over_5cm: usize,
    compensated_translation_over_10cm: usize,
    compensated_rotation_over_5deg: usize,
    compensated_rotation_over_10deg: usize,
}

impl VoDiagnosticStats {
    fn add_sample(
        &mut self,
        previous_camera_age: Duration,
        current_camera_age: Duration,
        measured_current_camera_to_previous_camera: nalgebra::Isometry3<f32>,
        previous_camera_matrix: &projection::camera_matrix::CameraMatrix,
        current_camera_matrix: &projection::camera_matrix::CameraMatrix,
    ) {
        let previous_robot_to_camera = robot_to_camera(previous_camera_matrix);
        let current_robot_to_camera = robot_to_camera(current_camera_matrix);
        let expected_current_camera_to_previous_camera =
            previous_robot_to_camera * current_robot_to_camera.inverse();
        let compensated_current_robot_to_previous_robot = previous_robot_to_camera.inverse()
            * measured_current_camera_to_previous_camera
            * current_robot_to_camera;
        let compensated_translation_norm = compensated_current_robot_to_previous_robot
            .translation
            .vector
            .norm();
        let compensated_rotation_angle = compensated_current_robot_to_previous_robot
            .rotation
            .angle()
            .to_degrees();
        let inverted_compensated_current_robot_to_previous_robot = previous_robot_to_camera
            .inverse()
            * measured_current_camera_to_previous_camera.inverse()
            * current_robot_to_camera;
        let measured_expected_error = expected_current_camera_to_previous_camera.inverse()
            * measured_current_camera_to_previous_camera;
        let measured_inverse_expected_error =
            expected_current_camera_to_previous_camera * measured_current_camera_to_previous_camera;
        push_vector_relation(
            &mut self.measured_expected_translation_cosine,
            &mut self.measured_expected_translation_ratio,
            measured_current_camera_to_previous_camera
                .translation
                .vector
                .cast(),
            expected_current_camera_to_previous_camera
                .translation
                .vector
                .cast(),
            1.0e-4,
        );
        push_vector_relation(
            &mut self.measured_expected_rotation_cosine,
            &mut self.measured_expected_rotation_ratio,
            measured_current_camera_to_previous_camera
                .rotation
                .scaled_axis()
                .cast(),
            expected_current_camera_to_previous_camera
                .rotation
                .scaled_axis()
                .cast(),
            1.0e-4,
        );

        self.samples += 1;
        self.compensated_translation_over_5cm += usize::from(compensated_translation_norm > 0.05);
        self.compensated_translation_over_10cm += usize::from(compensated_translation_norm > 0.10);
        self.compensated_rotation_over_5deg += usize::from(compensated_rotation_angle > 5.0);
        self.compensated_rotation_over_10deg += usize::from(compensated_rotation_angle > 10.0);
        self.previous_camera_age_ms
            .push(previous_camera_age.as_secs_f64() * 1000.0);
        self.current_camera_age_ms
            .push(current_camera_age.as_secs_f64() * 1000.0);
        push_isometry_norms(
            &mut self.measured_translation_m,
            &mut self.measured_rotation_deg,
            &measured_current_camera_to_previous_camera,
        );
        push_isometry_norms(
            &mut self.expected_translation_m,
            &mut self.expected_rotation_deg,
            &expected_current_camera_to_previous_camera,
        );
        push_isometry_norms(
            &mut self.compensated_translation_m,
            &mut self.compensated_rotation_deg,
            &compensated_current_robot_to_previous_robot,
        );
        push_isometry_norms(
            &mut self.inverted_compensated_translation_m,
            &mut self.inverted_compensated_rotation_deg,
            &inverted_compensated_current_robot_to_previous_robot,
        );
        push_isometry_norms(
            &mut self.measured_expected_error_translation_m,
            &mut self.measured_expected_error_rotation_deg,
            &measured_expected_error,
        );
        push_isometry_norms(
            &mut self.measured_inverse_expected_error_translation_m,
            &mut self.measured_inverse_expected_error_rotation_deg,
            &measured_inverse_expected_error,
        );
    }

    fn summary(&mut self) -> String {
        format!(
            "samples={} missing_camera_matrix={} compensated_trans_gt_5cm={} compensated_trans_gt_10cm={} compensated_rot_gt_5deg={} compensated_rot_gt_10deg={} prev_age_ms={} cur_age_ms={} measured_trans_m={} measured_rot_deg={} expected_trans_m={} expected_rot_deg={} compensated_trans_m={} compensated_rot_deg={} inverted_compensated_trans_m={} inverted_compensated_rot_deg={} measured_expected_error_trans_m={} measured_expected_error_rot_deg={} measured_inverse_expected_error_trans_m={} measured_inverse_expected_error_rot_deg={} measured_expected_translation_cosine={} measured_expected_translation_ratio={} measured_expected_rotation_cosine={} measured_expected_rotation_ratio={}",
            self.samples,
            self.missing_camera_matrix,
            self.compensated_translation_over_5cm,
            self.compensated_translation_over_10cm,
            self.compensated_rotation_over_5deg,
            self.compensated_rotation_over_10deg,
            quantiles(&mut self.previous_camera_age_ms),
            quantiles(&mut self.current_camera_age_ms),
            quantiles(&mut self.measured_translation_m),
            quantiles(&mut self.measured_rotation_deg),
            quantiles(&mut self.expected_translation_m),
            quantiles(&mut self.expected_rotation_deg),
            quantiles(&mut self.compensated_translation_m),
            quantiles(&mut self.compensated_rotation_deg),
            quantiles(&mut self.inverted_compensated_translation_m),
            quantiles(&mut self.inverted_compensated_rotation_deg),
            quantiles(&mut self.measured_expected_error_translation_m),
            quantiles(&mut self.measured_expected_error_rotation_deg),
            quantiles(&mut self.measured_inverse_expected_error_translation_m),
            quantiles(&mut self.measured_inverse_expected_error_rotation_deg),
            quantiles(&mut self.measured_expected_translation_cosine),
            quantiles(&mut self.measured_expected_translation_ratio),
            quantiles(&mut self.measured_expected_rotation_cosine),
            quantiles(&mut self.measured_expected_rotation_ratio),
        )
    }
}

fn push_isometry_norms(
    translations: &mut Vec<f64>,
    rotations: &mut Vec<f64>,
    isometry: &nalgebra::Isometry3<f32>,
) {
    translations.push(isometry.translation.vector.norm() as f64);
    rotations.push(isometry.rotation.angle().to_degrees() as f64);
}

fn push_vector_relation(
    cosines: &mut Vec<f64>,
    ratios: &mut Vec<f64>,
    measured: nalgebra::Vector3<f64>,
    expected: nalgebra::Vector3<f64>,
    min_expected_norm: f64,
) {
    let measured_norm = measured.norm();
    let expected_norm = expected.norm();
    if measured_norm <= 0.0 || expected_norm < min_expected_norm {
        return;
    }
    cosines.push(measured.dot(&expected) / (measured_norm * expected_norm));
    ratios.push(measured_norm / expected_norm);
}

fn quantiles(values: &mut [f64]) -> String {
    if values.is_empty() {
        return "n=0".to_string();
    }
    values.sort_by(|a, b| a.total_cmp(b));
    format!(
        "p50={:.4} p95={:.4} max={:.4}",
        percentile(values, 0.50),
        percentile(values, 0.95),
        values[values.len() - 1]
    )
}

fn percentile(values: &[f64], percentile: f64) -> f64 {
    let index = ((values.len() - 1) as f64 * percentile).round() as usize;
    values[index]
}

fn print_resolve_summary(
    recording: &mcap_recording::Recording,
    parameters: ReplayParameters,
    metrics_skip_seconds: f64,
) -> Result<()> {
    let replay = resolve_recording(recording, parameters)?;
    println!(
        "recorded_localization {}",
        trajectory_metrics_with_skip(
            &recording.recorded_localization_trajectory(),
            metrics_skip_seconds,
        )
    );
    println!(
        "replayed_localization {}",
        trajectory_metrics_with_skip(&replay.trajectory(), metrics_skip_seconds)
    );
    println!(
        "replay_stats imu_ingested={} foot_heights_ingested={} vo_received={} vo_ingested={} vo_dropped_invalid={} vo_dropped_gated={} vo_stale_camera_skips={} global_candidates={} global_frames_ingested={} global_associations_ingested={} global_unique_frames_ingested={} global_unique_associations_ingested={} pose_hint_frames_ingested={} pose_hint_associations_ingested={} solve_samples={}",
        replay.stats.imu_ingested,
        replay.stats.foot_heights_ingested,
        replay.stats.vo_received,
        replay.stats.vo_ingested,
        replay.stats.vo_dropped_invalid,
        replay.stats.vo_dropped_gated,
        replay.stats.vo_skipped_stale_camera_matrix,
        replay.stats.global_candidates,
        replay.stats.global_frames_ingested,
        replay.stats.global_associations_ingested,
        replay.stats.global_unique_frames_ingested,
        replay.stats.global_unique_associations_ingested,
        replay.stats.pose_hint_frames_ingested,
        replay.stats.pose_hint_associations_ingested,
        replay.samples.len(),
    );
    Ok(())
}

fn print_orientation_only_summary(recording: &mcap_recording::Recording) -> Result<()> {
    let parameters = ReplayParameters {
        include_visual_odometry: false,
        include_global_association: false,
        include_pose_hint_association: false,
        include_foot_heights: false,
        include_imu: true,
        ..ReplayParameters::default()
    };
    let replay = resolve_recording(recording, parameters)?;
    println!(
        "recorded_localization {}",
        trajectory_metrics(&recording.recorded_localization_trajectory())
    );
    println!(
        "orientation_only_replayed_localization {}",
        trajectory_metrics(&replay.trajectory())
    );
    println!(
        "replay_stats imu_ingested={} vo_received={} vo_ingested={} foot_heights_ingested={} global_candidates={} global_frames_ingested={} global_associations_ingested={} solve_samples={}",
        replay.stats.imu_ingested,
        replay.stats.vo_received,
        replay.stats.vo_ingested,
        replay.stats.foot_heights_ingested,
        replay.stats.global_candidates,
        replay.stats.global_frames_ingested,
        replay.stats.global_associations_ingested,
        replay.samples.len(),
    );
    Ok(())
}

fn robot_to_camera(
    camera_matrix: &projection::camera_matrix::CameraMatrix,
) -> nalgebra::Isometry3<f32> {
    (camera_matrix.head_to_camera * camera_matrix.robot_to_head).inner
}

fn trajectory_metrics(trajectory: &[TrajectoryPoint]) -> String {
    trajectory_metrics_with_skip(trajectory, 5.0)
}

fn trajectory_metrics_with_skip(trajectory: &[TrajectoryPoint], skip_seconds: f64) -> String {
    let Some(first) = trajectory.first() else {
        return "samples=0".to_string();
    };
    let skip_seconds = skip_seconds.max(0.0);
    let samples = trajectory
        .iter()
        .filter(|sample| sample.seconds - first.seconds >= skip_seconds)
        .collect::<Vec<_>>();
    let samples = if samples.is_empty() {
        trajectory.iter().collect::<Vec<_>>()
    } else {
        samples
    };
    let first = samples[0];
    let last = samples[samples.len() - 1];
    let xs = samples
        .iter()
        .map(|sample| sample.robot_to_field.inner.translation.vector.x)
        .collect::<Vec<_>>();
    let ys = samples
        .iter()
        .map(|sample| sample.robot_to_field.inner.translation.vector.y)
        .collect::<Vec<_>>();
    let yaws = unwrapped_yaws(&samples);
    let x_range = range(&xs);
    let y_range = range(&ys);
    let net_xy = ((last.robot_to_field.inner.translation.vector.x
        - first.robot_to_field.inner.translation.vector.x)
        .powi(2)
        + (last.robot_to_field.inner.translation.vector.y
            - first.robot_to_field.inner.translation.vector.y)
            .powi(2))
    .sqrt();
    let yaw_range = range(&yaws).to_degrees();
    let yaw_net = (yaws[yaws.len() - 1] - yaws[0]).to_degrees();

    format!(
        "samples={} window={:.2}-{:.2}s x_range={:.3}m y_range={:.3}m net_xy={:.3}m yaw_range={:.2}deg yaw_net={:.2}deg",
        samples.len(),
        first.seconds,
        last.seconds,
        x_range,
        y_range,
        net_xy,
        yaw_range,
        yaw_net,
    )
}

fn unwrapped_yaws(samples: &[&TrajectoryPoint]) -> Vec<f64> {
    let mut yaws = Vec::with_capacity(samples.len());
    for sample in samples {
        let mut yaw = sample.robot_to_field.inner.rotation.euler_angles().2;
        if let Some(previous) = yaws.last().copied() {
            while yaw - previous > std::f64::consts::PI {
                yaw -= std::f64::consts::TAU;
            }
            while yaw - previous < -std::f64::consts::PI {
                yaw += std::f64::consts::TAU;
            }
        }
        yaws.push(yaw);
    }
    yaws
}

fn range(values: &[f64]) -> f64 {
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    max - min
}

fn wgpu_options() -> WgpuConfiguration {
    let mut options = WgpuConfiguration::default();
    if let WgpuSetup::CreateNew(setup) = &mut options.wgpu_setup {
        let previous = setup.device_descriptor.clone();
        setup.device_descriptor = Arc::new(move |adapter| {
            let mut descriptor = previous(adapter);
            descriptor
                .required_limits
                .max_storage_buffers_per_shader_stage = 9;
            descriptor
        });
    }
    options
}

fn nearest_by_distance<T, Distance>(
    previous: Option<(T, Distance)>,
    next: Option<(T, Distance)>,
) -> Option<T>
where
    Distance: PartialOrd,
{
    match (previous, next) {
        (Some((previous, previous_distance)), Some((next, next_distance))) => {
            Some(if previous_distance <= next_distance {
                previous
            } else {
                next
            })
        }
        (Some((previous, _)), None) => Some(previous),
        (None, Some((next, _))) => Some(next),
        (None, None) => None,
    }
}
