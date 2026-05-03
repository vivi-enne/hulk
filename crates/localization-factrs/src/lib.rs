use crate::{
    backend::{BackendConfiguration, VinsBackend},
    frontend::VinsFrontend,
};

pub mod backend;
mod camera_intrinsics;
mod frontend;
pub mod imu_factor;
mod landmark_factor;
pub mod measurements;
pub mod node;
pub mod sparse_gaussian_process;
pub mod symbols;

pub fn initialize(config: BackendConfiguration) -> (VinsFrontend, VinsBackend) {
    let (measurement_sender, measurement_receiver) = tokio::sync::mpsc::unbounded_channel();
    let (result_sender, result_receiver) = tokio::sync::watch::channel(None);

    let frontend = VinsFrontend::new(measurement_sender, result_receiver);
    let backend = VinsBackend::new(config, measurement_receiver, result_sender);
    (frontend, backend)
}
