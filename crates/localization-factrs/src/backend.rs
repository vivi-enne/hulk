use std::time::{Duration, SystemTime};

use factrs::{
    assign_symbols,
    core::{GaussNewton, Graph, Values, Vector3},
    fac,
    linalg::Matrix3,
    optimizers::OptError,
    traits::Optimizer,
    variables::SE23,
};
use itertools::Itertools;
use thiserror::Error;

use crate::{imu_factor::IntervalGaussianProcessImuFactor, measurements::SensorMeasurement};

use tokio::sync::{
    mpsc::{UnboundedReceiver, error::TryRecvError},
    watch,
};

assign_symbols!(
    State: SE23;
);

pub struct BackendConfiguration {
    /// The spacing between control knots on the Gaussian Process
    /// Each control knot represents 9 DoFs for the optimizer.
    pub knot_spacing: Duration,
    /// The maximum optimization window size.
    /// Factors before the optimization window are marginalized.
    pub max_optimization_window: Duration,

    pub gyroscope_noise: Matrix3<f64>,
    pub accelerometer_noise: Matrix3<f64>,
    pub gravity: Vector3<f64>,
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
    config: BackendConfiguration,
    /// Stores the optimizer and the optimization graph
    optimizer: GaussNewton,
    /// Stores the optimized graph values.
    values: Option<Values>,
    /// Stores the timestamp of the last knot added to the graph.
    last_knot_time: Option<SystemTime>,
    start_time: SystemTime,
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
            config,
            optimizer: GaussNewton::new_default(Graph::default()),
            values: None,
            last_knot_time: None,
            start_time: SystemTime::now(),
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
        let mut current_measurements = Vec::new();
        loop {
            match self.measurement_receiver.try_recv() {
                Ok(SensorMeasurement::Imu(imu_measurement)) => {
                    current_measurements.push(imu_measurement)
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => return Err(FrontendDisconnected),
            }
        }
        current_measurements.sort_unstable_by_key(|measurement| measurement.time);

        // collect all measurements in 200ms blocks
        for (key, chunk) in &current_measurements
            .into_iter()
            .chunk_by(|imu_measurement| {
                get_previous_interval_start_time(
                    imu_measurement.time,
                    self.config.knot_spacing,
                    self.start_time,
                )
            })
        {
            let residual = IntervalGaussianProcessImuFactor::new(
                chunk.collect(),
                self.config.gyroscope_noise,
                self.config.accelerometer_noise,
                self.config.gravity,
                key,
                key + self.config.knot_spacing,
            );
            let interval_index =
                get_interval_index(key, self.config.knot_spacing, self.start_time) as u32;
            let factor = fac![residual, (State(interval_index), State(interval_index + 1))];
            self.optimizer.graph_mut().add_factor(factor);
        }

        Ok(())
    }

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

fn get_previous_interval_start_time(
    measurement_time: SystemTime,
    interval: Duration,
    start_time: SystemTime,
) -> SystemTime {
    let interval_index = get_interval_index(measurement_time, interval, start_time);
    let interval = interval.as_nanos();

    let previous_interval_start_time = interval_index * interval;
    start_time + Duration::from_nanos_u128(previous_interval_start_time)
}

fn get_interval_index(
    measurement_time: SystemTime,
    interval: Duration,
    start_time: SystemTime,
) -> u128 {
    let measurement_time = measurement_time
        .duration_since(start_time)
        .expect("time ran backwards")
        .as_nanos();
    let interval = interval.as_nanos();

    measurement_time / interval
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_get_previous_interval_start_time() {
        let start_time = SystemTime::now();
        let interval = Duration::from_millis(200);

        // 1. Middle of an interval
        // 265ms since epoch should snap back to 200ms
        let t1 = start_time + Duration::from_millis(265);
        let expected1 = start_time + interval;
        assert_eq!(
            get_previous_interval_start_time(t1, interval, start_time),
            expected1
        );

        // 2. Exact boundary
        // 200ms since epoch should stay at 200ms
        let t2 = start_time + Duration::from_millis(200);
        let expected2 = start_time + interval;
        assert_eq!(
            get_previous_interval_start_time(t2, interval, start_time),
            expected2
        );

        // 3. Just before a boundary
        // 399ms since epoch should snap back to 200ms
        let t3 = start_time + Duration::from_millis(399);
        let expected3 = start_time + interval;
        assert_eq!(
            get_previous_interval_start_time(t3, interval, start_time),
            expected3
        );

        // 4. Very early time
        // 50ms since epoch with 100ms interval should snap to 0 (Unix Epoch)
        let t4 = start_time + Duration::from_millis(50);
        let expected4 = start_time;
        assert_eq!(
            get_previous_interval_start_time(t4, interval, start_time),
            expected4
        );
    }

    #[test]
    fn test_large_intervals() {
        let start_time = SystemTime::now();

        // 1-second intervals
        let interval = Duration::from_secs(1);

        // 10.9 seconds -> 10.0 seconds
        let t = start_time + Duration::from_millis(10900);
        let expected = start_time + Duration::from_secs(10);
        assert_eq!(
            get_previous_interval_start_time(t, interval, start_time),
            expected,
        );
    }

    #[test]
    #[should_panic(expected = "time ran backwards")]
    fn test_pre_epoch_panic() {
        let start_time = SystemTime::now();

        // Optional: Ensure your expect() triggers if someone passes a time before 1970
        // (Only possible if the system clock is messed up or using a custom SystemTime)
        let way_back_when = start_time - Duration::from_secs(10);
        get_previous_interval_start_time(way_back_when, Duration::from_secs(1), start_time);
    }
}
