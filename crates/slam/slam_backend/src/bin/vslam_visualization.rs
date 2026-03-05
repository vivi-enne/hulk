use std::{
    collections::{BTreeMap, HashMap},
    fs::File,
    io::BufWriter,
};

use apex_solver::{
    CameraModel, LieGroup, PinholeCamera, SE3,
    camera_models::{DistortionModel, PinholeParams},
    core::problem::VariableEnum,
};
use color_eyre::Result;
use nalgebra::{Isometry3, Point3, Vector3};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::Serialize;
use slam::backend::{map::LandmarkMap, slam_backend::Backend};

#[derive(Serialize, Debug)]
struct PoseStep {
    true_pose: (f64, f64, f64),
    optimized_pose: (f64, f64, f64),
}

#[derive(Serialize, Debug)]
struct OutputData {
    // This provides the "history" key your plotting file expects
    history: Vec<PoseStep>,
}

#[derive(Serialize, Clone, Debug)]
struct LandmarkStep {
    lm_id: u64,
    true_pos: [f64; 3],
    opt_pos: [f64; 3],
}

fn main() -> Result<()> {
    let mut backend = Backend::new();
    let mut map = LandmarkMap::new();

    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()
        .ok();

    // Define a simple Pinhole Camera (fx, fy, cx, cy)
    let pinhole = PinholeParams::new(500.0, 500.0, 320.0, 240.0)?;
    let distortion = DistortionModel::None;
    let camera = PinholeCamera::new(pinhole, distortion)?;

    let mut true_poses = BTreeMap::new();
    let mut true_landmarks = BTreeMap::new();

    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let threshold = 0.1;

    // 1. Create a "Cloud" of True Landmarks
    for i in 0..100 {
        let pt = Point3::new(
            rng.random_range(-10.0..10.0),
            rng.random_range(-10.0..10.0),
            rng.random_range(0.0..20.0),
        );
        true_landmarks.insert(i as u64, pt);
    }
    dbg!(&true_landmarks);

    let mut prev_true_t_cw: Option<Isometry3<f64>> = None;
    let mut prev_guess_t_cw: Option<Isometry3<f64>> = None;
    let mut prev_id: Option<u64> = None;

    // 2. Spiral Trajectory Simulation
    for i in 0..200 {
        let t = i as f64 * 0.2;
        let x = t.cos() * 2.0;
        let y = t.sin() * 2.0;
        let z = t * 0.5; // Upward movement

        let true_iso = Isometry3::translation(x, y, z);
        let true_t_cw = true_iso.inverse();

        let current_guess_t_cw;
        let pose_id;

        // let pose_id = backend.add_pose(SE3::from_isometry(t_cw.clone()));
        // true_poses.insert(pose_id, Point3::from(true_iso.translation.vector));

        // // Add Odometry constraint expecting T_wc variables
        // if let (Some(p_t_wc), Some(p_id)) = (&prev_t_wc, prev_id) {
        //     // Relative motion: T_prev^{-1} * T_curr
        //     let relative_delta = p_t_wc.inverse() * true_iso.clone();
        //     backend.add_between(p_id, pose_id, SE3::from_isometry(relative_delta));
        // }
        // prev_t_wc = Some(true_iso.clone());
        // prev_id = Some(pose_id);
        // Odometry between T_cw frames
        if let (Some(p_true_t_cw), Some(p_guess_t_cw), Some(p_id)) =
            (&prev_true_t_cw, &prev_guess_t_cw, prev_id)
        {
            let true_delta = p_true_t_cw.inverse() * true_t_cw.clone();

            // 2. Inject realistic Gaussian noise (translation and small rotation)
            let noise_scale = 0.01;
            let t_noise = Vector3::new(
                rng.random_range(-noise_scale..noise_scale),
                rng.random_range(-noise_scale..noise_scale),
                rng.random_range(-noise_scale..noise_scale),
            );

            let axis = Vector3::new(
                rng.random_range(-1.0..1.0),
                rng.random_range(-1.0..1.0),
                rng.random_range(-1.0..1.0),
            )
            .normalize();
            let angle = rng.random_range(-0.02..0.02);
            let r_noise = nalgebra::UnitQuaternion::from_axis_angle(
                &nalgebra::Unit::new_normalize(axis),
                angle,
            );

            let noise_iso = Isometry3::from_parts(t_noise.into(), r_noise);

            // Apply the noise to the true relative motion
            let noisy_delta = true_delta * noise_iso;

            // 3. Accumulate the noisy delta to get our realistic initial guess
            current_guess_t_cw = p_guess_t_cw * noisy_delta.clone();

            // 4. Feed the noisy guess to the solver
            pose_id = backend.add_pose(SE3::from_isometry(current_guess_t_cw.clone()));

            // 5. Provide the noisy odometry measurement as the BetweenFactor
            backend.add_between(p_id, pose_id, SE3::from_isometry(noisy_delta));
        } else {
            // First pose acts as our perfect anchor (Zero-state)
            current_guess_t_cw = true_t_cw.clone();
            pose_id = backend.add_pose(SE3::from_isometry(current_guess_t_cw.clone()));
        }
        true_poses.insert(pose_id, Point3::from(true_iso.translation.vector));

        prev_true_t_cw = Some(true_t_cw.clone());
        prev_guess_t_cw = Some(current_guess_t_cw);
        prev_id = Some(pose_id);

        // prev_t_cw = Some(t_cw);
        // prev_id = Some(pose_id);

        let mut frame_landmarks = Vec::new();
        for (id, &pt) in &true_landmarks {
            // dbg!(id);
            let local_pt = true_iso.inverse() * pt;
            if local_pt.z > 0.5 && local_pt.z < 15.0 {
                let uv = camera.project(&local_pt.coords)?;
                let desc = vec![*id as f32; 32];

                let landmark_id = match map.find_match(&desc, threshold) {
                    Some(m_id) => m_id,
                    None => {
                        let noisy_pt = pt + Vector3::new(rng.random_range(-0.2..0.2), 0.0, 0.0);
                        map.add_landmark(noisy_pt, desc);
                        backend.add_landmark_variable(*id, noisy_pt);
                        *id
                    }
                };
                // dbg!(pose_id, landmark_id);
                backend.add_projection(pose_id, landmark_id, uv, camera.clone());
                frame_landmarks.push(landmark_id);
            } else {
                // dbg!(id, pt, local_pt);
            }
        }
        // dbg!(frame_landmarks);
    }

    let result = backend.optimize();

    // 3. Extraction with Landmarks
    let mut history = Vec::new();
    let mut landmarks_out = Vec::new();

    for (name, var_enum) in &result.parameters {
        if name.starts_with('x') {
            let id: u64 = name[1..].parse()?;
            if let (VariableEnum::SE3(se3), Some(tp)) = (var_enum, true_poses.get(&id)) {
                // let op = se3.value.translation();
                let op = se3.value.inverse(None).translation();

                history.push((
                    id,
                    PoseStep {
                        true_pose: (tp.x, tp.y, tp.z),
                        // optimized_pose: (0.0, 0.0, 0.0), //(op.x, op.y, op.z),
                        optimized_pose: (op.x, op.y, op.z),
                    },
                ));
            }
        } else if name.starts_with('l') {
            let id: u64 = name[1..].parse()?;
            if let (VariableEnum::Rn(rn), Some(tp)) = (var_enum, true_landmarks.get(&id)) {
                let op = rn.value.data();
                landmarks_out.push(LandmarkStep {
                    lm_id: id,
                    true_pos: (tp.x, tp.y, tp.z).into(),
                    opt_pos: (op[0], op[1], op[2]).into(),
                });
            }
        }
    }

    dbg!(landmarks_out.clone());
    history.sort_by_key(|x| x.0);
    let final_history: Vec<PoseStep> = history.into_iter().map(|x| x.1).collect();

    let output = serde_json::json!({
        "history": final_history,
        "landmarks": landmarks_out
    });

    serde_json::to_writer(
        BufWriter::new(File::create("apex_pose_graph.json")?),
        &output,
    )?;
    Ok(())
}
