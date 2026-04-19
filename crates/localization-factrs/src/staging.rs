use std::time::SystemTime;

use booster::ImuState;

pub struct StagingArea {
    imu_measurements: Vec<(SystemTime, ImuState)>,
}

impl StagingArea {
    pub fn new() -> Self {
        Self {
            imu_measurements: Vec::new(),
        }
    }

    pub fn add_imu_measurement(&mut self, time: SystemTime, state: ImuState) {
        self.imu_measurements.push((time, state));
    }
}
