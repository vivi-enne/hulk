use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct SimulationData {
    pub start_pose: SimulatorState,
    pub landmark_global_positions: LandmarkGlobalPositions,
    pub measurements: Vec<SimulatorMeasurement>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LandmarkGlobalPositions {
    pub corner_top_left: [f64; 3],
    pub corner_top_right: [f64; 3],
    pub corner_bottom_left: [f64; 3],
    pub corner_bottom_right: [f64; 3],
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SimulatorMeasurement {
    pub timestamp_seconds: f64,
    pub imu: SimulatorImuMeasurement,
    pub ground_truth_pose: SimulatorState,
    pub visual_features: Option<SimulatorDetectedFeatures>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SimulatorImuMeasurement {
    pub angular_velocity: [f64; 3],
    pub linear_acceleration: [f64; 3],
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SimulatorState {
    pub position: [f64; 3],
    pub quaternion_wxyz: [f64; 4],
    pub linear_velocity_global: [f64; 3],
    pub angular_velocity_local: [f64; 3],
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SimulatorDetectedFeatures {
    pub detections: Vec<[f64; 2]>,
    pub intrinsics: CameraIntrinsics,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CameraIntrinsics {
    pub focal_x: f64,
    pub focal_y: f64,
    pub center_x: f64,
    pub center_y: f64,
}
