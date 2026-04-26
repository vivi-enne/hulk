use std::time::SystemTime;

use booster::ImuState;

pub enum SensorMeasurement {
    Imu(ImuMeasurement),
}

#[derive(Debug, Clone)]
pub struct ImuMeasurement {
    pub time: SystemTime,
    pub state: ImuState,
}
