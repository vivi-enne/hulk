use apex_solver::factors::Factor;
use nalgebra::{DMatrix, DVector};

pub struct WeightedFactor<F: Factor> {
    pub inner: F,
    pub weight: f64,
}

impl<F: Factor> Factor for WeightedFactor<F> {
    fn linearize(
        &self,
        params: &[DVector<f64>],
        compute_jacobian: bool,
    ) -> (DVector<f64>, Option<DMatrix<f64>>) {
        // Get the raw pixel or metric errors from the apex-solver factor
        let (mut res, mut jac) = self.inner.linearize(params, compute_jacobian);
        
        // Scale the residual vector
        res *= self.weight;
        
        // Scale the Jacobian matrix to match
        if let Some(j) = &mut jac {
            *j *= self.weight;
        }
        
        (res, jac)
    }

    fn get_dimension(&self) -> usize {
        self.inner.get_dimension()
    }
}