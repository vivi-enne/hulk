use std::{ops::AddAssign, time::SystemTime};

use booster::ImuState;
use factrs::{
    core::Vector3,
    linalg::{ForwardProp, Matrix3, Numeric, VectorX},
    residuals::Residual2,
    traits::Variable,
    variables::{MatrixLieGroup, SE23},
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
    type DimOut = Const<6>;

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

        let mut residual = VectorX::<T>::zeros(6);
        for measurement in &self.measurements {
            let current_pose = spline.infer(measurement.time);
            let body_frame_gravity = current_pose
                .rot()
                .inverse()
                .apply(self.gravity.cast::<T>().as_view());

            let inferred_derivative = spline.infer_derivative(measurement.time);
            let predicted_gyro = inferred_derivative.fixed_view::<3, 1>(0, 0);
            let predicted_accel = inferred_derivative.fixed_view::<3, 1>(3, 0) + body_frame_gravity;

            let gyro_residual =
                predicted_gyro - measurement.state.angular_velocity.inner.cast::<T>();
            let accel_residual =
                predicted_accel - measurement.state.linear_acceleration.inner.cast::<T>();

            residual
                .fixed_view_mut::<3, 1>(0, 0)
                .add_assign(&gyro_residual.map(|x| x * x));
            residual
                .fixed_view_mut::<3, 1>(3, 0)
                .add_assign(&accel_residual.map(|x| x * x));
        }

        residual
    }
}
