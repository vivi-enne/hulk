pub mod simulation_data;

use std::{
    path::PathBuf,
    time::{Duration, Instant, SystemTime},
};

use booster::ImuState;
use clap::Parser;
use color_eyre::{Result, eyre::Context};
use indicatif::ProgressIterator;
use linear_algebra::IntoFramed;
use localization_factrs::{backend::BackendConfiguration, initialize};
use nalgebra::{Matrix3, Vector3, vector};

use crate::simulation_data::SimulationData;

#[derive(Debug, Parser)]
struct Arguments {
    simulation: PathBuf,
}

pub fn main() -> Result<()> {
    let arguments = Arguments::parse();
    let data = std::fs::read_to_string(arguments.simulation)?;
    let data: SimulationData = serde_json::from_str(&data).wrap_err("failed to deserialize")?;

    let (mut frontend, mut backend) = initialize(BackendConfiguration {
        knot_spacing: Duration::from_millis(200),
        max_optimization_window: Duration::from_secs(3),
        gyroscope_noise: Matrix3::identity() * 0.01,
        accelerometer_noise: Matrix3::identity() * 0.01,
        gravity: vector![0.0, 0.0, 9.81],
    });
    let now = SystemTime::now();

    for frame in data.measurements.iter().progress() {
        let angular_velocity = Vector3::from(frame.imu.angular_velocity)
            .cast::<f32>()
            .framed();
        let linear_acceleration = Vector3::from(frame.imu.linear_acceleration)
            .cast::<f32>()
            .framed();

        frontend.ingest_imu(
            now + Duration::from_secs_f64(frame.timestamp_seconds),
            ImuState {
                roll_pitch_yaw: vector![0.0, 0.0, 0.0].framed(),
                angular_velocity,
                linear_acceleration,
            },
        )?;
        backend.solve_once()?;
    }

    Ok(())
}
