use std::time::SystemTime;

use booster::ImuState;
use factrs::{
    core::Vector3, linalg::{ForwardProp, Matrix3, Numeric, VectorX}, residuals::Residual2, traits::Variable, variables::SE23
};
use nalgebra::Const;

use crate::sparse_gaussian_process::SE23SparseGaussianProcessSegment;

#[derive(Debug, Clone)]
pub struct IntervalGaussianProcessImuFactor {
    pub measurements: Vec<ImuMeasurement>,
    pub gyroscope_noise: Matrix3<f64>,
    pub accelerometer_noise: Matrix3<f64>,
    pub gravity: Vector3<f64>,
    pub start_time: SystemTime,
    pub end_time: SystemTime,
}

#[derive(Debug, Clone)]
pub struct ImuMeasurement {
    pub time: SystemTime,
    pub state: ImuState,
}

impl IntervalGaussianProcessImuFactor {}

#[factrs::mark]
impl Residual2 for IntervalGaussianProcessImuFactor {
    type V1 = SE23;
    type V2 = SE23;

    type DimIn = Const<18>;
    type DimOut = Const<9>;

    type Differ = ForwardProp<Const<18>>;

    fn residual2<T: Numeric>(&self, pose_start: SE23<T>, pose_end: SE23<T>) -> VectorX<T> {
        let spline = SE23SparseGaussianProcessSegment::new(
            self.start_time,
            pose_start,
            self.end_time,
            pose_end,
            self.gyroscope_noise,
            self.accelerometer_noise,
        );

        for measurement in &self.measurements {
            let inferred = spline.infer(measurement.time);
            let derivative = inferred.

            
        }

        todo!()
    }
}
