use std::time::SystemTime;

use factrs::{linalg::Numeric, traits::Variable, variables::SE23};
use nalgebra::{Matrix3, SMatrix, SVector};

pub struct SE23SparseGaussianProcessSegment<T: Numeric> {
    start_time: SystemTime,
    start_pose: SE23<T>,
    interpolation_factor: SVector<T, 9>,
    gyro_noise: Matrix3<f64>,
    accelerometer_noise: Matrix3<f64>,
}

impl<T: Numeric> SE23SparseGaussianProcessSegment<T> {
    pub fn new(
        start_time: SystemTime,
        start_pose: SE23<T>,
        end_time: SystemTime,
        end_pose: SE23<T>,
        gyro_noise: Matrix3<f64>,
        accelerometer_noise: Matrix3<f64>,
    ) -> Self {
        let duration = end_time
            .duration_since(start_time)
            .expect("invalid time order")
            .as_secs_f64();
        let covariance_between =
            Self::noise_covariance(&gyro_noise, &accelerometer_noise, duration).cast::<T>();

        let relative_algebra_error = start_pose.inverse().compose(&end_pose).log();
        let relative_algebra_error: SVector<T, 9> =
            SVector::from_column_slice(relative_algebra_error.as_slice());
        let interpolation_factor = covariance_between
            .cholesky()
            .expect("covariance must be positive definite")
            .solve(&relative_algebra_error);

        SE23SparseGaussianProcessSegment {
            start_time,
            start_pose,
            interpolation_factor,
            gyro_noise,
            accelerometer_noise,
        }
    }

    /// Interpolates the state at a specific timestamp.
    ///
    /// This method computes the expected state at time $\tau$ using the Gaussian Process
    /// prior. It calculates the interpolation error in the local tangent space and applies
    /// it to the starting pose.
    ///
    /// The tangent space error is mathematically defined as
    /// $\mathbf{e}(\Delta t) = \mathbf{Q}(\Delta t) \boldsymbol{\Phi}(\Delta t)^\top \boldsymbol{\lambda}$.
    /// To maximize performance, the $9 \times 9$ matrix multiplications are bypassed entirely
    /// by analytically expanding the sparse block matrices into independent $3 \times 1$ vectors:
    ///
    /// - **Orientation:** $\mathbf{e}_0 = \Delta t \boldsymbol{\Sigma}_g \boldsymbol{\lambda}_0$
    /// - **Velocity:** $\mathbf{e}_1 = \Delta t \boldsymbol{\Sigma}_a \boldsymbol{\lambda}_1 + 1.5 \Delta t^2 \boldsymbol{\Sigma}_a \boldsymbol{\lambda}_2$
    /// - **Position:** $\mathbf{e}_2 = 0.5 \Delta t^2 \boldsymbol{\Sigma}_a \boldsymbol{\lambda}_1 + \frac{5}{6} \Delta t^3 \boldsymbol{\Sigma}_a \boldsymbol{\lambda}_2$
    ///
    /// The final pose is recovered via the right retraction operator: $T(\tau) = T_{\text{start}} \oplus \mathbf{e}(\Delta t)$.
    #[allow(non_snake_case)]
    pub fn infer(&self, tau: SystemTime) -> SE23<T> {
        let duration_previous = tau
            .duration_since(self.start_time)
            .expect("segment must start before timestamp");

        let delta_time = duration_previous.as_secs_f64();
        let delta_time_squared = delta_time * delta_time;
        let delta_time_cubed = delta_time_squared * delta_time;

        let lambda_0 = self.interpolation_factor.fixed_view::<3, 1>(0, 0);
        let lambda_1 = self.interpolation_factor.fixed_view::<3, 1>(3, 0);
        let lambda_2 = self.interpolation_factor.fixed_view::<3, 1>(6, 0);

        // Scale prior to casting to ensure trait bounds are satisfied for multiplication
        let gyro_scaled_time = self.gyro_noise.scale(delta_time).cast::<T>();

        let accel_scaled_time = self.accelerometer_noise.scale(delta_time).cast::<T>();
        let accel_scaled_1_5_time_squared = self
            .accelerometer_noise
            .scale(1.5 * delta_time_squared)
            .cast::<T>();
        let accel_scaled_0_5_time_squared = self
            .accelerometer_noise
            .scale(0.5 * delta_time_squared)
            .cast::<T>();
        let accel_scaled_5_6_time_cubed = self
            .accelerometer_noise
            .scale((5.0 / 6.0) * delta_time_cubed)
            .cast::<T>();

        let error_0 = gyro_scaled_time * lambda_0;
        let error_1 = (accel_scaled_time * lambda_1) + (accel_scaled_1_5_time_squared * lambda_2);
        let error_2 =
            (accel_scaled_0_5_time_squared * lambda_1) + (accel_scaled_5_6_time_cubed * lambda_2);

        let mut error = SVector::<T, 9>::zeros();
        error.fixed_view_mut::<3, 1>(0, 0).copy_from(&error_0);
        error.fixed_view_mut::<3, 1>(3, 0).copy_from(&error_1);
        error.fixed_view_mut::<3, 1>(6, 0).copy_from(&error_2);

        self.start_pose.oplus_right(error.as_view())
    }

    /// Computes the analytical time derivative of the local tangent space error.
    ///
    /// This method calculates the rate of change of the algebra element $\mathbf{e}(t)$
    /// with respect to time, evaluated at time $\tau$. This is derived by applying the
    /// product rule to the underlying tangent error equation:
    ///
    /// $\frac{\partial \mathbf{e}}{\partial t} = \dot{\mathbf{Q}}(\Delta t) \boldsymbol{\Phi}(\Delta t)^\top \boldsymbol{\lambda} + \mathbf{Q}(\Delta t) \dot{\boldsymbol{\Phi}}(\Delta t)^\top \boldsymbol{\lambda}$
    ///
    /// Dense matrix allocations are avoided by resolving the sparse structures directly
    /// into $3 \times 1$ components:
    ///
    /// - **Orientation Rate:** $\dot{\mathbf{e}}_0 = \boldsymbol{\Sigma}_g \boldsymbol{\lambda}_0$
    /// - **Velocity Rate:** $\dot{\mathbf{e}}_1 = \boldsymbol{\Sigma}_a \boldsymbol{\lambda}_1 + 3 \Delta t \boldsymbol{\Sigma}_a \boldsymbol{\lambda}_2$
    /// - **Position Rate:** $\dot{\mathbf{e}}_2 = \Delta t \boldsymbol{\Sigma}_a \boldsymbol{\lambda}_1 + 2.5 \Delta t^2 \boldsymbol{\Sigma}_a \boldsymbol{\lambda}_2$
    pub fn infer_derivative(&self, tau: SystemTime) -> SVector<T, 9> {
        let duration_previous = tau
            .duration_since(self.start_time)
            .expect("segment must start before timestamp");

        let delta_time = duration_previous.as_secs_f64();
        let delta_time_squared = delta_time * delta_time;

        let lambda_0 = self.interpolation_factor.fixed_view::<3, 1>(0, 0);
        let lambda_1 = self.interpolation_factor.fixed_view::<3, 1>(3, 0);
        let lambda_2 = self.interpolation_factor.fixed_view::<3, 1>(6, 0);

        let gyro_noise_t = self.gyro_noise.cast::<T>();
        let accelerometer_noise_t = self.accelerometer_noise.cast::<T>();

        let accel_scaled_3_dt = self.accelerometer_noise.scale(3.0 * delta_time).cast::<T>();
        let accel_scaled_dt = self.accelerometer_noise.scale(delta_time).cast::<T>();
        let accel_scaled_2_5_dt_sq = self
            .accelerometer_noise
            .scale(2.5 * delta_time_squared)
            .cast::<T>();

        let block_0 = gyro_noise_t * lambda_0;
        let block_1 = (accelerometer_noise_t * lambda_1) + (accel_scaled_3_dt * lambda_2);
        let block_2 = (accel_scaled_dt * lambda_1) + (accel_scaled_2_5_dt_sq * lambda_2);

        let mut derivative = SVector::<T, 9>::zeros();
        derivative.fixed_view_mut::<3, 1>(0, 0).copy_from(&block_0);
        derivative.fixed_view_mut::<3, 1>(3, 0).copy_from(&block_1);
        derivative.fixed_view_mut::<3, 1>(6, 0).copy_from(&block_2);

        derivative
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
        let gp = SE23SparseGaussianProcessSegment::new(
            t_start,
            SE23::from_rot_vel_trans(SO3::identity(), Vector3::zeros(), Vector3::zeros()),
            t_start + Duration::from_secs(1),
            SE23::from_rot_vel_trans(
                SO3::identity(),
                vector![2.0, 0.0, 0.0],
                vector![1.0, 0.0, 0.0],
            ),
            Matrix3::identity(),
            Matrix3::identity(),
        );

        dbg!(gp.infer(t_start + Duration::from_millis(500)));
    }
}
