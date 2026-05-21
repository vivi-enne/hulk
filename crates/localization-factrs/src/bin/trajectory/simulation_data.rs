use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct SimulationData {
    start_pose: SimulatorState,
    landmark_global_positions: LandmarkGlobalPositions,
    measurements: Vec<SimulatorMeasurement>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LandmarkGlobalPositions {
    corner_top_left: [f64; 3],
    corner_top_right: [f64; 3],
    corner_bottom_left: [f64; 3],
    corner_bottom_right: [f64; 3],
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SimulatorMeasurement {
    timestamp_seconds: f64,
    imu: SimulatorImuMeasurement,
    ground_truth_pose: SimulatorState,
    visual_features: Option<SimulatorDetectedFeatures>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SimulatorImuMeasurement {
    angular_velocity: [f64; 3],
    linear_acceleration: [f64; 3],
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SimulatorState {
    position: [f64; 3],
    quaternion_wxyz: [f64; 4],
    linear_velocity_global: [f64; 3],
    angular_velocity_local: [f64; 3],
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SimulatorDetectedFeatures {
    detections: Vec<[f64; 2]>,
    intrinsics: CameraIntrinsics,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CameraIntrinsics {
    focal_x: f64,
    focal_y: f64,
    center_x: f64,
    center_y: f64,
}
