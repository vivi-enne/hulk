use std::time::{Duration, SystemTime};

use factrs::{
    core::{Graph, Values},
    traits::Variable,
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
    measurement_receiver: UnboundedReceiver<SensorMeasurement>,
    result_sender: watch::Sender<Option<OptimizationResult>>,
    _config: BackendConfiguration,
    _graph: Graph,
    _values: Values,
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
            _graph: Graph::default(),
            _values: Values::default(),
        }
    }

    /// Loops continuously, ingesting measurements and optimizing the graph.
    /// Only returns if an error occurs.
    pub fn run_loop(mut self) -> Result<(), VinsBackendError> {
        loop {
            self.ingest_until_empty()?;

            let result = self.optimize();
            if self.result_sender.send(Some(result)).is_err() {
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

    fn optimize(&mut self) -> OptimizationResult {
        // Build task, marginalize, solve
        OptimizationResult {
            time: SystemTime::now(),
            latest_pose: SE23::identity(),
        }
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
