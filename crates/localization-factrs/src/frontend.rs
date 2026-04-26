use std::time::SystemTime;

use booster::ImuState;
use thiserror::Error;
use tokio::sync::{mpsc::UnboundedSender, watch};

use crate::{
    backend::OptimizationResult,
    measurements::{ImuMeasurement, SensorMeasurement},
};

pub struct VinsFrontend {
    measurement_sender: UnboundedSender<SensorMeasurement>,
    result_receiver: watch::Receiver<Option<OptimizationResult>>,
}

impl VinsFrontend {
    pub fn new(
        measurement_sender: UnboundedSender<SensorMeasurement>,
        result_receiver: watch::Receiver<Option<OptimizationResult>>,
    ) -> Self {
        Self {
            measurement_sender,
            result_receiver,
        }
    }

    pub fn last_optimization_result(&self) -> Option<OptimizationResult> {
        (*self.result_receiver.borrow()).clone()
    }

    /// Adds an IMU measurement to the optimization pipeline.
    pub fn ingest_imu(
        &mut self,
        time: SystemTime,
        state: ImuState,
    ) -> Result<(), VinsFrontendError> {
        self.measurement_sender
            .send(SensorMeasurement::Imu(ImuMeasurement { time, state }))
            .map_err(|_| VinsFrontendError::BackendDisconnected)
    }
}

#[derive(Debug, Error)]
pub enum VinsFrontendError {
    #[error("the localization backend is disconnected")]
    BackendDisconnected,
}
