use std::time::SystemTime;

use factrs::{traits::Variable, variables::SE23};
use nalgebra::{Matrix3, SMatrix, SVector};

pub struct PrecomputedSe23SparseGaussianProcess {
    gyro_noise: Matrix3<f64>,
    accelerometer_noise: Matrix3<f64>,
    segments: Vec<TrajectorySegment>,
}

struct TrajectorySegment {
    start_time: SystemTime,
    start_pose: SE23,
    interpolation_factor: SVector<f64, 9>,
}

impl PrecomputedSe23SparseGaussianProcess {
    pub fn new(
        gyro_noise: Matrix3<f64>,
        accelerometer_noise: Matrix3<f64>,
        control_points: &[(SystemTime, SE23)],
    ) -> Self {
        let segments = control_points
            .array_windows::<2>()
            .map(|[(t_prev, pose_prev), (t_after, pose_after)]| {
                Self::build_segment(
                    *t_prev,
                    pose_prev.clone(),
                    *t_after,
                    pose_after.clone(),
                    &gyro_noise,
                    &accelerometer_noise,
                )
            })
            .collect();

        Self {
            gyro_noise,
            accelerometer_noise,
            segments,
        }
    }

    fn build_segment(
        start_time: SystemTime,
        start_pose: SE23,
        end_time: SystemTime,
        end_pose: SE23,
        gyro_noise: &Matrix3<f64>,
        accelerometer_noise: &Matrix3<f64>,
    ) -> TrajectorySegment {
        let duration = end_time
            .duration_since(start_time)
            .expect("invalid time order")
            .as_secs_f64();
        let covariance_between = Self::noise_covariance(gyro_noise, accelerometer_noise, duration);

        let relative_algebra_error = start_pose.inverse().compose(&end_pose).log();
        let relative_algebra_error: SVector<f64, 9> =
            SVector::from_column_slice(relative_algebra_error.as_slice());
        let interpolation_factor = covariance_between
            .cholesky()
            .expect("covariance must be positive definite")
            .solve(&relative_algebra_error);

        TrajectorySegment {
            start_time,
            start_pose,
            interpolation_factor,
        }
    }

    #[allow(non_snake_case)]
    pub fn infer(&self, tau: SystemTime) -> SE23 {
        let segment = match self
            .segments
            .binary_search_by(|segment| segment.start_time.cmp(&tau))
        {
            // Exact match found
            Ok(index) => return self.segments[index].start_pose.clone(),
            // Inside the GP support
            Err(index) if index > 0 => &self.segments[index - 1],
            // Out of support
            Err(_) => todo!(),
        };

        let duration_prev = tau
            .duration_since(*&segment.start_time)
            .expect("segment must start before timestamp");
        let Phi_prev = Self::transition_matrix(duration_prev.as_secs_f64());
        let Q_prev = Self::noise_covariance(
            &self.gyro_noise,
            &self.accelerometer_noise,
            duration_prev.as_secs_f64(),
        );

        let projected = Phi_prev.transpose() * segment.interpolation_factor;
        let error = Q_prev * projected;

        segment.start_pose.oplus_right(error.as_view())
    }

    #[allow(non_snake_case)]
    fn transition_matrix(dt: f64) -> SMatrix<f64, 9, 9> {
        let mut Phi = SMatrix::<f64, 9, 9>::identity();
        Phi.fixed_view_mut::<3, 3>(6, 3)
            .copy_from(&Matrix3::identity().scale(dt));
        Phi
    }

    #[allow(non_snake_case)]
    fn noise_covariance(
        gyro_noise: &Matrix3<f64>,
        accelerometer_noise: &Matrix3<f64>,
        dt: f64,
    ) -> SMatrix<f64, 9, 9> {
        let mut Q_d = SMatrix::<f64, 9, 9>::zeros();
        Q_d.fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&gyro_noise.scale(dt));
        Q_d.fixed_view_mut::<3, 3>(3, 3)
            .copy_from(&accelerometer_noise.scale(dt));
        Q_d.fixed_view_mut::<3, 3>(3, 6)
            .copy_from(&accelerometer_noise.scale(dt * dt / 2.));
        Q_d.fixed_view_mut::<3, 3>(6, 3)
            .copy_from(&accelerometer_noise.scale(dt * dt / 2.));
        Q_d.fixed_view_mut::<3, 3>(6, 6)
            .copy_from(&accelerometer_noise.scale(dt * dt * dt / 3.));
        Q_d
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use factrs::core::{SO3, Vector3};
    use nalgebra::vector;

    use super::*;

    #[test]
    fn test_infer() {
        let t_start = SystemTime::now();
        let gp = PrecomputedSe23SparseGaussianProcess::new(
            Matrix3::identity(),
            Matrix3::identity(),
            &[
                (
                    t_start,
                    SE23::from_rot_vel_trans(SO3::identity(), Vector3::zeros(), Vector3::zeros()),
                ),
                (
                    t_start + Duration::from_secs(1),
                    SE23::from_rot_vel_trans(
                        SO3::identity(),
                        vector![2.0, 0.0, 0.0],
                        vector![1.0, 0.0, 0.0],
                    ),
                ),
            ],
        );

        dbg!(gp.infer(t_start + Duration::from_millis(500)));
    }
}
