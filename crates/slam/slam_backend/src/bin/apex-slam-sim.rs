use color_eyre::Result;
use indicatif::ProgressBar;
use nalgebra::{Isometry3, UnitQuaternion, Vector3};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};
use serde::Serialize;

use apex_solver::{
    core::problem::Problem,
    factors::{BetweenFactor, PriorFactor},
    linalg::LinearSolverType,
    manifold::se3::SE3,
    optimizer::levenberg_marquardt::{LevenbergMarquardt, LevenbergMarquardtConfig},
};

use std::collections::HashMap;
use std::fs::File;
use std::io::BufWriter;

#[derive(Serialize)]
struct PoseStep {
    true_pose: (f64, f64, f64),
    optimized_pose: (f64, f64, f64),
}

#[derive(Serialize)]
struct OutputData {
    history: Vec<PoseStep>,
}

fn main() -> Result<()> {
    const DT: f64 = 0.1;
    const SIM_TIME: f64 = 60.0;

    let mut rng = ChaCha8Rng::seed_from_u64(42);

    let trans_noise = Normal::new(0.0, 0.02)?;
    let rot_noise = Normal::new(0.0, 0.01)?;

    let mut problem = Problem::new();
    let mut initial_values = HashMap::new();

    let mut true_pose = Isometry3::identity();
    let mut last_noisy_pose = Isometry3::identity();

    let mut history = Vec::new();
    let mut last_id = 0;

    let bar = ProgressBar::new((SIM_TIME / DT) as u64);

    // Add first pose with prior
    {
        let se3 = SE3::from_isometry(true_pose);
        let var = format!("x0");

        let dv = nalgebra::dvector![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];

        initial_values.insert(var.clone(), (apex_solver::ManifoldType::SE3, dv));
        let prior = PriorFactor { data: se3.into() };
        problem.add_residual_block(&[&var], Box::new(prior), None);
    }

    for step in 1..((SIM_TIME / DT) as usize) {
        bar.inc(1);

        let time = step as f64 * DT;

        let (v_forward, v_up, yaw_rate) = if time <= 1.0 {
            (0.0, 0.0, 0.0)
        } else {
            (1.0, 0.05, 0.1)
        };

        let delta_t = Vector3::new(v_forward * DT, 0.0, v_up * DT);
        let delta_r = UnitQuaternion::from_euler_angles(0.0, 0.0, yaw_rate * DT);

        let motion = Isometry3::from_parts(delta_t.into(), delta_r);

        // Update ground truth
        true_pose = true_pose * motion;

        // Add noise to odometry
        let noisy_trans = delta_t
            + Vector3::new(
                trans_noise.sample(&mut rng),
                trans_noise.sample(&mut rng),
                trans_noise.sample(&mut rng),
            );

        let noisy_rot = UnitQuaternion::from_euler_angles(
            rot_noise.sample(&mut rng),
            rot_noise.sample(&mut rng),
            rot_noise.sample(&mut rng),
        );

        let noisy_motion = Isometry3::from_parts(noisy_trans.into(), delta_r * noisy_rot);

        last_noisy_pose = last_noisy_pose * noisy_motion;

        let var_prev = format!("x{}", last_id);
        let var_curr = format!("x{}", last_id + 1);


        let dv = nalgebra::dvector![
            last_noisy_pose.translation.vector.x,
            last_noisy_pose.translation.vector.y,
            last_noisy_pose.translation.vector.z,
            last_noisy_pose.rotation.quaternion().w,
            last_noisy_pose.rotation.quaternion().i,
            last_noisy_pose.rotation.quaternion().j,
            last_noisy_pose.rotation.quaternion().k,
        ];

        initial_values.insert(var_curr.clone(), (apex_solver::ManifoldType::SE3, dv));

        let between = BetweenFactor::<SE3>::new(SE3::from_isometry(noisy_motion));
        problem.add_residual_block(&[&var_prev, &var_curr], Box::new(between), None);

        last_id += 1;
    }

    bar.finish();

    // Optimize
    let config = LevenbergMarquardtConfig::new()
        .with_linear_solver_type(LinearSolverType::SparseCholesky)
        .with_max_iterations(50);

    let mut solver = LevenbergMarquardt::with_config(config);
    let result = solver.optimize(&problem, &initial_values)?;

    // Extract optimized poses
    for i in 0..=last_id {
        let var = format!("x{}", i);
        if let Some(opt) = result.parameters.get(&var) {
            // dbg!("Optimized {}: {:?}", var, opt);

            let translation = opt.to_vector();
            // dbg!("Translation: {:?}", translation.clone());
            let x = translation[0];
            let y = translation[1];
            let z = translation[2];

            let se3 = SE3::from_isometry(true_pose);
            let true_translation = se3.translation();

            history.push(PoseStep {
                true_pose: (
                    true_translation[0],
                    true_translation[1],
                    true_translation[2],
                ), // could store separately
                optimized_pose: (x, y, z),
            });
        }
    }

    let output = OutputData { history };

    let file = File::create("apex_pose_graph.json")?;
    let writer = BufWriter::new(file);
    serde_json::to_writer(writer, &output)?;

    println!("Written apex_pose_graph.json");

    Ok(())
}
