use std::time::Duration;

use booster::ImuState;
use factrs::{traits::Variable, variables::SE23};
use nalgebra::{SMatrix, vector};

pub struct PreintegratedImuGpFactor {
    pub relative_measurement: SE23,
    pub information_matrix_sqrt: SMatrix<f64, 9, 9>,
}

pub struct ImuMeasurement {
    pub dt: Duration,
    pub state: ImuState,
}

impl PreintegratedImuGpFactor {
    pub fn build(measurements: &[ImuMeasurement], gp_covariance: &SMatrix<f64, 9, 9>) -> Self {
        let relative_measurement =
            measurements
                .into_iter()
                .fold(SE23::identity(), |acc, ImuMeasurement { dt, state }| {
                    let twist = vector![
                        state.angular_velocity.x() as f64, // Angular rate
                        state.angular_velocity.y() as f64,
                        state.angular_velocity.z() as f64,
                        state.linear_acceleration.x() as f64, // Velocity rate (Acceleration)
                        state.linear_acceleration.y() as f64,
                        state.linear_acceleration.z() as f64,
                        0.0, // Translation rate
                        0.0,
                        0.0,
                    ] * dt.as_secs_f64();
                    acc.oplus_right(twist.as_view())
                });

        let covariance_cholesky = gp_covariance
            .cholesky()
            .expect("covariance matrix must be positive definite");

        let information_matrix_sqrt = covariance_cholesky
            .l()
            .try_inverse()
            .expect("lower triangular matrix must be invertible");

        Self {
            relative_measurement,
            information_matrix_sqrt,
        }
    }
}

// impl Residual2 for PreintegratedImuGpFactor {
//     type V1 = SE23;
//     type V2 = SE23;

//     type DimIn = Const<18>;
//     type DimOut = Const<9>;

//     type Differ = ForwardProp<Const<18>>;

//     fn residual2<T: Numeric>(&self, pose_start: SE23<T>, pose_end: SE23<T>) -> VectorX<T> {
//         let expected_pose_end = pose_start.compose(&self.relative_measurement);
//         let local_error = expected_pose_end.inverse().compose(&pose_end).log();
//         self.information_matrix_sqrt * local_error
//     }
// }
