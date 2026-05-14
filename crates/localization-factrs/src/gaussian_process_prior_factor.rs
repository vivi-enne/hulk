use factrs::{
    containers::Key,
    core::Values,
    linalg::{MatrixX, Numeric, VectorX},
    traits::{Diff, Residual, Variable},
    variables::{MatrixLieGroup, SE23},
};
use nalgebra::{SMatrix, SVector};

use crate::sparse_gaussian_process::SE23SparseGaussianProcessSegment;

#[derive(Debug, Clone)]
pub struct GaussianProcessPriorFactor {
    duration: f64,
    information_root: SMatrix<f64, 9, 9>,
}

impl GaussianProcessPriorFactor {
    pub fn new(
        duration: f64,
        gyroscope_noise: &nalgebra::Matrix3<f64>,
        accelerometer_noise: &nalgebra::Matrix3<f64>,
    ) -> Self {
        let covariance_matrix = SE23SparseGaussianProcessSegment::<f64>::noise_covariance(
            gyroscope_noise,
            accelerometer_noise,
            duration,
        );

        let information_root = covariance_matrix
            .cholesky()
            .expect("covariance must be positive definite")
            .l()
            .try_inverse()
            .expect("lower triangular matrix must be invertible");

        Self {
            duration,
            information_root,
        }
    }

    fn residual<T: Numeric>(&self, start_pose: &SE23<T>, end_pose: &SE23<T>) -> VectorX<T> {
        let start_rotation_inverse = start_pose.rot().inverse();
        let local_velocity = start_rotation_inverse.apply(start_pose.uvw());

        let mut predicted_tangent = SVector::<T, 9>::zeros();
        predicted_tangent
            .fixed_view_mut::<3, 1>(6, 0)
            .copy_from(&local_velocity.scale(T::from(self.duration)));

        let predicted_end_pose = start_pose.oplus_right(predicted_tangent.as_view());

        let relative_algebra_error = predicted_end_pose.inverse().compose(end_pose).log();
        let relative_algebra_error_vector =
            SVector::<T, 9>::from_column_slice(relative_algebra_error.as_slice());

        let mut residual = VectorX::<T>::zeros(9);
        residual.copy_from(&(self.information_root.cast::<T>() * relative_algebra_error_vector));
        residual
    }
}

impl Residual for GaussianProcessPriorFactor {
    fn num_keys(&self) -> usize {
        2
    }

    fn dim_in(&self) -> usize {
        2 * SE23::<f64>::DIM
    }

    fn dim_out(&self) -> usize {
        9
    }

    fn residual(&self, values: &Values, keys: &[Key]) -> VectorX {
        let start_pose = values.get_unchecked::<_, SE23>(keys[0]).unwrap();
        let end_pose = values.get_unchecked::<_, SE23>(keys[1]).unwrap();

        self.residual(start_pose, end_pose)
    }

    fn residual_jacobian(
        &self,
        values: &Values,
        keys: &[Key],
    ) -> factrs::linalg::DiffResult<VectorX, MatrixX> {
        let start_pose = values.get_unchecked::<_, SE23>(keys[0]).unwrap();
        let end_pose = values.get_unchecked::<_, SE23>(keys[1]).unwrap();

        factrs::linalg::ForwardProp::<nalgebra::Const<18>>::jacobian_2(
            |start, end| self.residual(&start, &end),
            start_pose,
            end_pose,
        )
    }
}
