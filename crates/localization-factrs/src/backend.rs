use std::time::{Duration, SystemTime};

use factrs::{
    core::{GaussNewton, Graph, Values},
    optimizers::OptError,
    traits::Optimizer,
    variables::SE23,
};
use thiserror::Error;

use crate::measurements::SensorMeasurement;

use tokio::sync::{
    mpsc::{UnboundedReceiver, error::TryRecvError},
    watch,
};

pub struct BackendConfiguration {
    /// The spacing between control knots on the Gaussian Process
    /// Each control knot represents 9 DoFs for the optimizer.
    pub knot_spacing: Duration,
    /// The maximum optimization window size.
    /// Factors before the optimization window are marginalized.
    pub max_optimization_window: Duration,
}

#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// The timestamp of the most recent variable in the optimized graph
    pub time: SystemTime,
    /// The latest pose estimate from the optimized graph
    pub latest_pose: SE23<f64>,
}

pub struct VinsBackend {
    /// Channel to retrieve new measurements from the frontend.
    measurement_receiver: UnboundedReceiver<SensorMeasurement>,
    /// Channel to send solver results to the frontend,
    result_sender: watch::Sender<Option<OptimizationResult>>,
    /// Configuration parameters for the solver backend
    _config: BackendConfiguration,
    /// Stores the optimizer and the optimization graph
    optimizer: GaussNewton,
    /// Stores the optimized graph values.
    values: Option<Values>,
    /// Stores the timestamp of the last knot added to the graph.
    last_knot_time: Option<SystemTime>,
}

impl VinsBackend {
    pub(crate) fn new(
        config: BackendConfiguration,
        measurement_receiver: UnboundedReceiver<SensorMeasurement>,
        result_sender: watch::Sender<Option<OptimizationResult>>,
    ) -> Self {
        Self {
            measurement_receiver,
            result_sender,
            _config: config,
            optimizer: GaussNewton::new_default(Graph::default()),
            values: None,
            last_knot_time: None,
        }
    }

    /// Loops continuously, ingesting measurements and optimizing the graph.
    /// Only returns if an error occurs.
    pub fn run_loop(mut self) -> Result<(), VinsBackendError> {
        loop {
            self.ingest_until_empty()?;

            let result = self.optimize();
            if self.result_sender.send(result).is_err() {
                return Err(VinsBackendError::FrontendDisconnected(FrontendDisconnected));
            }
        }
    }

    fn ingest_until_empty(&mut self) -> Result<(), FrontendDisconnected> {
        loop {
            match self.measurement_receiver.try_recv() {
                Ok(measurement) => self.process_measurement(measurement),
                Err(TryRecvError::Empty) => return Ok(()),
                Err(TryRecvError::Disconnected) => return Err(FrontendDisconnected),
            }
        }
    }

    fn process_measurement(&mut self, _measurement: SensorMeasurement) {}

    fn optimize(&mut self) -> Option<OptimizationResult> {
        let values = self.values.take().unwrap_or_default();
        let result = self.optimizer.optimize(values);

        match result {
            Ok(values) => {
                self.values = Some(values);
            }
            Err(OptError::MaxIterations(values)) => {
                log::warn!("optimizer failed to converge: max iterations reached");
                self.values = Some(values);
            }
            Err(OptError::FailedToStep) => {
                log::warn!("optimizer failed: failed to step");
            }
            Err(OptError::InvalidSystem) => {
                log::warn!("optimizer failed: invalid system");
            }
        };

        let values = self.values.as_ref()?;
        let time = self.last_knot_time?;
        let latest_pose = values.filter::<SE23<f64>>().last()?.clone();

        Some(OptimizationResult { time, latest_pose })
    }
}

#[derive(Debug, Error)]
#[error("frontend disconnected")]
pub struct FrontendDisconnected;

#[derive(Debug, Error)]
pub enum VinsBackendError {
    #[error(transparent)]
    FrontendDisconnected(#[from] FrontendDisconnected),
}
