// apex-solver pose-graph backend
use apex_solver::{
    BundleAdjustment, ManifoldType, PinholeCamera, ProjectionFactor,
    core::problem::{Problem, VariableEnum},
    factors::{BetweenFactor, PriorFactor},
    linalg::LinearSolverType,
    manifold::se3::SE3,
    optimizer::{
        SolverResult,
        levenberg_marquardt::{LevenbergMarquardt, LevenbergMarquardtConfig},
    },
};
use color_eyre::Result;
use nalgebra::{DVector, Matrix2xX, Point3, Vector2};
use std::collections::HashMap;

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

        // encode as [tx,ty,tz,qw,qx,qy,qz]
        // let dv = nalgebra::dvector![
        //     pose.translation().x,
        //     pose.translation().y,
        //     pose.translation().z,
        //     pose.rotation_so3().quaternion().w,
        //     pose.rotation_so3().quaternion().i,
        //     pose.rotation_so3().quaternion().j,
        //     pose.rotation_so3().quaternion().k,
        // ];

        // // encode as [tx, ty, tz, qx, qy, qz, qw]
        // let dv = nalgebra::dvector![
        //     pose.translation().x,
        //     pose.translation().y,
        //     pose.translation().z,
        //     pose.rotation_so3().quaternion().i, // x
        //     pose.rotation_so3().quaternion().j, // y
        //     pose.rotation_so3().quaternion().k, // z
        //     pose.rotation_so3().quaternion().w, // w (scalar last)
        // ];

        let dv: DVector<f64> = pose.into();
        self.initial_values.insert(
            var_name.clone(),
            (apex_solver::ManifoldType::SE3, dv.clone()),
        );

        // add a weak prior on the first pose
        if id == 0 {
            // let prior = PriorFactor { data: pose.into() };
            let prior = PriorFactor { data: dv };
            self.problem
                .add_residual_block(&[&var_name], Box::new(prior), None);
        }

        id
    }

    pub fn add_between(&mut self, i: u64, j: u64, relative: SE3) {
        let from = format!("x{}", i);
        let to = format!("x{}", j);

        let factor = BetweenFactor::<SE3>::new(relative);
        self.problem
            .add_residual_block(&[&from, &to], Box::new(factor), None);
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
    ) {
        let pose_var = format!("x{}", pose_id);
        let lm_var = format!("l{}", landmark_id);

        let dyn_measurement = Matrix2xX::from_columns(&[measurement]);

        // Initialize the native apex_solver ProjectionFactor
        let factor: ProjectionFactor<PinholeCamera, BundleAdjustment> =
            ProjectionFactor::new(dyn_measurement, camera);

        // Add to the problem graph connecting the specific pose and landmark
        self.problem.add_residual_block(
            &[&pose_var, &lm_var],
            Box::new(factor),
            None, // Optional robust loss function (e.g., Huber) can go here
        );
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
