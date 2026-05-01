use std::time::{Duration, SystemTime};

use booster::ImuState;
use color_eyre::Result;
use context_attribute::context;
use factrs::{core::Vector3, linalg::Matrix3, traits::Variable, variables::SE23};
use framework::{AdditionalOutput, PerceptionInput, deserialize_not_implemented};
use hardware::{CameraInterface, TimeInterface};
use nalgebra::{Isometry3, Quaternion, Translation3, UnitQuaternion, vector};
use serde::{Deserialize, Serialize};

use crate::{backend::BackendConfiguration, frontend::VinsFrontend, initialize};

#[derive(Deserialize, Serialize)]
pub struct Localization {
    time: SystemTime,
    #[serde(skip, default = "deserialize_not_implemented")]
    state: SE23,
    #[serde(skip, default = "deserialize_not_implemented")]
    frontend: VinsFrontend,
}

#[context]
pub struct CreationContext {}

#[context]
pub struct CycleContext {
    hardware_interface: HardwareInterface,
    imu_state: PerceptionInput<ImuState, "Motion", "imu_state">,
    dead_reckoning: AdditionalOutput<Isometry3<f32>, "dead_reckoning">,
}

#[context]
pub struct MainOutputs {}

impl Localization {
    pub fn new(_context: CreationContext) -> Result<Self> {
        let (frontend, backend) = initialize(BackendConfiguration {
            knot_spacing: Duration::from_millis(200),
            max_optimization_window: Duration::from_secs(3),
            // TODO: check in documentation of booster robot
            gyroscope_noise: Matrix3::identity() * 0.01,
            accelerometer_noise: Matrix3::identity() * 0.1,
            gravity: Vector3::new(0., 0., 9.81),
        });
        std::thread::spawn(move || {
            backend
                .run_loop()
                .expect("localization backend closed unexpectedly")
        });
        Ok(Self {
            time: SystemTime::UNIX_EPOCH,
            frontend,
            state: SE23::identity(),
        })
    }

    pub fn cycle(
        &mut self,
        mut context: CycleContext<impl CameraInterface + TimeInterface>,
    ) -> Result<MainOutputs> {
        for (time, imus) in context.imu_state.persistent {
            if let Some(imu) = imus.last() {
                self.frontend.ingest_imu(time, **imu)?;

                let dt = time.duration_since(self.time).unwrap_or_default();
                let twist = vector![
                    imu.angular_velocity.x() as f64, // Angular rate
                    imu.angular_velocity.y() as f64,
                    imu.angular_velocity.z() as f64,
                    imu.linear_acceleration.x() as f64, // Velocity rate (Acceleration)
                    imu.linear_acceleration.y() as f64,
                    imu.linear_acceleration.z() as f64,
                    0.0, // Translation rate
                    0.0,
                    0.0,
                ] * dt.as_secs_f64();

                self.state = self.state.oplus_right(twist.as_view());
                self.time = time;
            }
        }

        // self.frontend.last_optimization_result().map(|result| {
        //     result.latest_pose
        // })

        context
            .dead_reckoning
            .fill_if_subscribed(|| se23_to_isometry3(self.state.clone()));

        Ok(MainOutputs {})
    }
}

fn se23_to_isometry3(pose: SE23) -> Isometry3<f32> {
    let xyz = pose.xyz();
    let rot = pose.rot();

    Isometry3::from_parts(
        Translation3::from(xyz.into_owned().cast::<f32>()),
        UnitQuaternion::from_quaternion(Quaternion::from_vector(rot.xyzw.cast::<f32>())),
    )
}
