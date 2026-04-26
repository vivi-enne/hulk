use std::time::{Duration, SystemTime};

use booster::ImuState;
use color_eyre::Result;
use context_attribute::context;
use factrs::{traits::Variable, variables::SE23};
use framework::{AdditionalOutput, PerceptionInput};
use hardware::{CameraInterface, TimeInterface};
use nalgebra::{Isometry3, Quaternion, Translation3, UnitQuaternion, vector};
use serde::{Deserialize, Serialize};
use types::object_detection::{Detection, NaoLabelPartyObjectDetectionLabel};

use crate::{backend::BackendConfiguration, initialize};

#[derive(Deserialize, Serialize)]
pub struct ImageReceiver {
    time: SystemTime,
    #[serde(skip)]
    state: State,
}

struct State {
    state: SE23,
}

impl Default for State {
    fn default() -> Self {
        Self {
            state: SE23::identity(),
        }
    }
}

#[context]
pub struct CreationContext {}

#[context]
pub struct CycleContext {
    hardware_interface: HardwareInterface,
    imu_state: PerceptionInput<ImuState, "Motion", "imu_state">,
    object_detections: PerceptionInput<
        Vec<Detection<NaoLabelPartyObjectDetectionLabel>>,
        "ObjectDetection",
        "detected_objects",
    >,

    dead_reckoning: AdditionalOutput<Isometry3<f32>, "dead_reckoning">,
}

#[context]
pub struct MainOutputs {}

impl ImageReceiver {
    pub fn new(_context: CreationContext) -> Result<Self> {
        let (frontend, backend) = initialize(BackendConfiguration {
            knot_spacing: Duration::from_millis(200),
            max_optimization_window: Duration::from_secs(3),
        });
        std::thread::spawn(move || {
            backend
                .run_loop()
                .expect("localization backend closed unexpectedly")
        });
        Ok(Self {
            time: SystemTime::UNIX_EPOCH,
            state: State::default(),
        })
    }

    pub fn cycle(
        &mut self,
        mut context: CycleContext<impl CameraInterface + TimeInterface>,
    ) -> Result<MainOutputs> {
        for (time, imus) in context.imu_state.persistent {
            if let Some(imu) = imus.last() {
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

                self.state.state = self.state.state.oplus_right(twist.as_view());
                self.time = time;
            }
        }

        context.dead_reckoning.fill_if_subscribed(|| {
            Isometry3::from_parts(
                Translation3::from(
                    vector![
                        self.state.state.xyz().x,
                        self.state.state.xyz().y,
                        self.state.state.xyz().z,
                    ]
                    .cast::<f32>(),
                ),
                UnitQuaternion::from_quaternion(Quaternion::from_vector(
                    self.state.state.rot().xyzw.cast::<f32>(),
                )),
            )
        });

        Ok(MainOutputs {})
    }
}
