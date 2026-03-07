// apex-solver pose-graph backend
use apex_solver::{
    BundleAdjustment, ManifoldType, PinholeCamera, ProjectionFactor,
    core::{
        loss_functions::{HuberLoss, LossFunction},
        problem::{Problem, VariableEnum},
    },
    factors::{BetweenFactor, PriorFactor},
    linalg::LinearSolverType,
    manifold::se3::SE3,
    optimizer::{
        SolverResult,
        levenberg_marquardt::{LevenbergMarquardt, LevenbergMarquardtConfig},
    },
};
use nalgebra::{DVector, Matrix2xX, Point3, Vector2};
use std::collections::HashMap;

use crate::backend::weighted_factor::WeightedFactor;

/// A simple pose graph backend
pub struct Backend {
    problem: Problem,
    initial_values: HashMap<String, (ManifoldType, DVector<f64>)>,
    last_id: u64,
}

impl Backend {
    pub fn new() -> Self {
        Self {
            problem: Problem::new(),
            initial_values: HashMap::new(),
            last_id: 0,
        }
    }

    pub fn add_pose(&mut self, pose: SE3) -> u64 {
        let id = self.last_id;
        self.last_id += 1;

        let var_name = format!("x{}", id);

        let dv: DVector<f64> = pose.into();
        self.initial_values.insert(
            var_name.clone(),
            (apex_solver::ManifoldType::SE3, dv.clone()),
        );

        // add a weak prior on the first pose
        if id == 0 {
            // let prior = PriorFactor { data: pose.into() };
            let prior = PriorFactor { data: dv.clone() };
            // self.problem
            //     .add_residual_block(&[&var_name], Box::new(prior), None);

            // 1000.0 weight acts as an immovable anchor
            let hard_anchor = WeightedFactor {
                inner: prior,
                weight: 1000.0,
            };

            self.problem
                .add_residual_block(&[&var_name], Box::new(hard_anchor), None);
        }

        id
    }

    pub fn add_between(&mut self, i: u64, j: u64, relative: SE3, weight: f64) {
        let from = format!("x{}", i);
        let to = format!("x{}", j);

        let factor = BetweenFactor::<SE3>::new(relative);
        // self.problem
        //     .add_residual_block(&[&from, &to], Box::new(factor), None);

        // Use the WeightedFactor wrapper
        let weighted = WeightedFactor {
            inner: factor,
            weight,
        };
        self.problem
            .add_residual_block(&[&from, &to], Box::new(weighted), None);
    }

    /// Register a 3D landmark as a variable to be optimized
    pub fn add_landmark_variable(&mut self, landmark_id: u64, initial_position: Point3<f64>) {
        let var_name = format!("l{}", landmark_id);

        let dv = nalgebra::dvector![initial_position.x, initial_position.y, initial_position.z];

        // Landmarks are typically optimized in standard 3D Euclidean space
        self.initial_values.insert(var_name, (ManifoldType::RN, dv));
    }

    /// Add a projection constraint (Visual Measurement)
    pub fn add_projection(
        &mut self,
        pose_id: u64,
        landmark_id: u64,
        measurement: Vector2<f64>,
        camera: PinholeCamera,
        weight: f64,
    ) {
        let pose_var = format!("x{}", pose_id);
        let lm_var = format!("l{}", landmark_id);

        let dyn_measurement = Matrix2xX::from_columns(&[measurement]);

        // Initialize the native apex_solver ProjectionFactor
        let factor: ProjectionFactor<PinholeCamera, BundleAdjustment> =
            ProjectionFactor::new(dyn_measurement, camera);

        let weighted_factor = WeightedFactor {
            inner: factor,
            weight,
        };

        // Scale the Huber threshold to match the new weighted residual
        let huber = Box::new(HuberLoss::new(2.0).expect("Huber loss initialization failed"));

        self.problem.add_residual_block(
            &[&pose_var, &lm_var],
            Box::new(weighted_factor),
            Some(huber),
        );
    }

    /// Anchors a landmark heavily to fix the Gauge Freedom (Coordinate Frame)
    pub fn add_landmark_prior(&mut self, landmark_id: u64, position: Point3<f64>) {
        let var_name = format!("l{}", landmark_id);
        let dv = nalgebra::dvector![position.x, position.y, position.z];

        // Add a strong weight by duplicating the factor (acts as a stiff anchor)
        for _ in 0..100 {
            let prior = PriorFactor { data: dv.clone() };
            self.problem
                .add_residual_block(&[&var_name], Box::new(prior), None);
        }
    }
    /// Adds a weak prior to prevent singular matrices for distant, unconstrained landmarks
    pub fn add_weak_landmark_prior(
        &mut self,
        landmark_id: u64,
        position: Point3<f64>,
        weight: f64,
    ) {
        let var_name = format!("l{}", landmark_id);
        let dv = nalgebra::dvector![position.x, position.y, position.z];

        // We use your WeightedFactor wrapper to make this constraint very loose
        let prior = PriorFactor { data: dv };
        let weighted_prior = WeightedFactor {
            inner: prior,
            weight,
        };

        self.problem
            .add_residual_block(&[&var_name], Box::new(weighted_prior), None);
    }

    pub fn optimize(&mut self) -> SolverResult<HashMap<String, VariableEnum>> {
        let config = LevenbergMarquardtConfig::new()
            .with_linear_solver_type(LinearSolverType::SparseCholesky)
            .with_max_iterations(50);
        let mut solver = LevenbergMarquardt::with_config(config);

        let result = solver
            .optimize(&self.problem, &self.initial_values)
            .expect("SLAM solver did not find a solution");
        result
    }
}
