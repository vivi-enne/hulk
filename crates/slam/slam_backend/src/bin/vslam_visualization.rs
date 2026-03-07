use std::{
    collections::{BTreeMap, HashMap},
    fs::File,
    io::BufWriter,
};

use apex_solver::{
    CameraModel, LieGroup, PinholeCamera, SE3,
    camera_models::{DistortionModel, PinholeParams},
    core::problem::VariableEnum,
    optimizer::SolverResult,
};
use color_eyre::Result;
use nalgebra::{Const, Isometry3, OPoint, Point3, Vector3};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::Serialize;
use slam_backend::backend::{map::LandmarkMap, slam_backend::Backend};

#[derive(Serialize, Debug)]
struct PoseStep {
    true_pose: (f64, f64, f64),
    optimized_pose: (f64, f64, f64),
}

#[derive(Serialize, Debug)]
struct OutputData {
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
    // let mut true_landmarks = BTreeMap::new();

    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let threshold = 0.1;

    // 1. Create a "Cloud" of True Landmarks
    let true_landmarks = generate_true_landmarks(50, &mut rng);

    dbg!(&true_landmarks);

    let x = 2.0;
    let y = 0.0;
    let z = 0.0; // Upward movement
    let true_iso = Isometry3::translation(x, y, z);

    // First pose acts as our perfect anchor (Zero-state)
    let current_guess_t_cw = true_iso.inverse().clone();
    let pose_id = backend.add_pose(SE3::from_isometry(current_guess_t_cw.clone()));
    true_poses.insert(pose_id, Point3::from(true_iso.translation.vector));

    let mut prev_true_t_cw = true_iso.inverse();
    let mut prev_guess_t_cw = current_guess_t_cw;
    let mut prev_id = pose_id;

    process_landmarks_for_pose(
        &true_landmarks,
        &true_iso,
        &camera,
        &mut map,
        &mut backend,
        pose_id,
        threshold,
        &mut rng,
    )?;

    // 2. Spiral Trajectory Simulation
    for i in 0..200 {
        let t = i as f64 * 0.2;
        let x = t.cos() * 2.0;
        let y = t.sin() * 2.0;
        let z = t * 0.5; // Upward movement

        let true_iso = Isometry3::translation(x, y, z);

        //// start working
        let t_cw = true_iso.inverse();
        let pose_id = backend.add_pose(SE3::from_isometry(t_cw.clone()));
        true_poses.insert(pose_id, Point3::from(true_iso.translation.vector));

        if i > 0 {
            // Odometry between T_cw frames
            let relative_delta = prev_true_t_cw.inverse() * t_cw.clone();
            backend.add_between(prev_id, pose_id, SE3::from_isometry(relative_delta));
        }

        prev_true_t_cw = t_cw;
        prev_id = pose_id;


        process_landmarks_for_pose(
            &true_landmarks,
            &true_iso,
            &camera,
            &mut map,
            &mut backend,
            pose_id,
            threshold,
            &mut rng,
        )?;
    }

    let result = backend.optimize();

    let path = "apex_pose_graph.json";
    extract_and_save_result(result, true_poses, true_landmarks, path)?;

    Ok(())
}

fn generate_true_landmarks(count: usize, rng: &mut impl Rng) -> BTreeMap<u64, Point3<f64>> {
    (0..count as u64)
        .map(|i| {
            let pt = Point3::new(
                rng.random_range(-10.0..10.0),
                rng.random_range(-10.0..10.0),
                rng.random_range(0.0..20.0),
            );
            (i, pt)
        })
        .collect()
}

fn process_landmarks_for_pose(
    landmarks: &BTreeMap<u64, Point3<f64>>,
    camera_isometry: &Isometry3<f64>,
    camera: &PinholeCamera,
    map: &mut LandmarkMap,
    backend: &mut Backend,
    pose_id: u64,
    threshold: f32,
    rng: &mut ChaCha8Rng,
) -> Result<()> {
    let mut frame_landmarks = Vec::new();
    for (id, &true_landmark_position) in landmarks {
        let local_pt = camera_isometry.inverse() * true_landmark_position;
        if local_pt.z > 0.5 && local_pt.z < 15.0 {
            let uv = camera.project(&local_pt.coords)?;
            let desc = vec![*id as f32; 32];

            let landmark_id = match map.find_match(&desc, threshold) {
                Some(m_id) => m_id,
                None => {
                    let noisy_pt = true_landmark_position
                        + Vector3::new(rng.random_range(-0.2..0.2), 0.0, 0.0);
                    map.add_landmark(noisy_pt, desc);
                    backend.add_landmark_variable(*id, noisy_pt);
                    *id
                }
            };
            backend.add_projection(pose_id, landmark_id, uv, camera.clone());
            frame_landmarks.push(landmark_id);
        }
    }

    Ok(())
}

fn extract_and_save_result(
    result: SolverResult<HashMap<String, VariableEnum>>,
    true_poses: BTreeMap<u64, OPoint<f64, Const<3>>>,
    true_landmarks: BTreeMap<u64, OPoint<f64, Const<3>>>,
    path: &str,
) -> Result<()> {
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

    history.sort_by_key(|x| x.0);
    let final_history: Vec<PoseStep> = history.into_iter().map(|x| x.1).collect();

    let output = serde_json::json!({ "history": final_history, "landmarks": landmarks_out });
    let file = File::create(path)?;
    serde_json::to_writer(BufWriter::new(file), &output)?;
    Ok(())
}
