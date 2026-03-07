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
use nalgebra::{Const, Isometry3, OPoint, Point3, Point4, Vector2, Vector3};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};
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

    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let threshold = 0.1;

    let true_landmarks = generate_true_landmarks(150, &mut rng);

    dbg!(&true_landmarks);

    let x = 5.0;
    let y = 0.0;
    let z = 0.0; // Upward movement
    let true_iso = Isometry3::translation(x, y, z);

    // First pose acts as our perfect anchor (Zero-state)
    let mut current_guess_t_cw = true_iso.inverse().clone();
    let mut pose_id = backend.add_pose(SE3::from_isometry(current_guess_t_cw.clone()));
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
    let total_steps = 200;
    for i in 1..total_steps {
        // let t = i as f64 * 0.2;
        // let x = t.cos() * 2.0;
        // let y = t.sin() * 2.0;
        // let z = t * 0.5; // Upward movement

        // Create a full circle (2 * PI) over the 200 steps
        let t = (i as f64) * (2.0 * std::f64::consts::PI / (total_steps - 20) as f64);

        // Radius of 5.0 meters, flat on the Z-plane
        let x = t.cos() * 5.0;
        let y = t.sin() * 5.0;
        let z = 0.0;

        // To make the camera "look" around the room, we also rotate it
        // to face tangent to the circle
        let rotation =
            nalgebra::UnitQuaternion::from_euler_angles(0.0, t + std::f64::consts::FRAC_PI_2, 0.0);
        let translation = nalgebra::Translation3::new(x, y, z);
        let true_iso = Isometry3::from_parts(translation, rotation);

        let true_t_cw = true_iso.inverse();

        // Odometry between T_cw frames
        let true_relative_delta = prev_true_t_cw.inverse() * true_t_cw.clone();

        let trans_std = 0.1; // 1cm standard deviation
        let rot_std = 0.005; // ~0.05 degree standard deviation
        let trans_dist = Normal::new(0.0, trans_std)?;
        let rot_dist = Normal::new(0.0, rot_std)?;

        let noisy_delta =
            apply_gaussian_noise_to_isometry(true_relative_delta, &trans_dist, &rot_dist, &mut rng);

        // WEIGHT: 1 / variance
        let odom_sigma = 0.02;
        let odom_weight = 1.0 / (odom_sigma * odom_sigma); // Weight = 2500.0

        let px_sigma = 1.0;
        let px_weight = 1.0 / (px_sigma * px_sigma); // Weight = 1.0

        // accumulate the noisy delta to get our realistic initial guess
        current_guess_t_cw = prev_guess_t_cw * noisy_delta.clone();

        // Feed noisy guess to the solver
        pose_id = backend.add_pose(SE3::from_isometry(current_guess_t_cw.clone()));

        backend.add_between(
            prev_id,
            pose_id,
            SE3::from_isometry(noisy_delta),
            odom_weight,
        );

        true_poses.insert(pose_id, Point3::from(true_iso.translation.vector));

        prev_true_t_cw = true_t_cw;
        prev_guess_t_cw = current_guess_t_cw;
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
    let px_weight = 1.0;

    for (id, &true_landmark_position) in landmarks {
        let local_pt = camera_isometry.inverse() * true_landmark_position;
        if local_pt.z > 0.5 && local_pt.z < 20.0 {
            let uv = camera.project(&local_pt.coords)?;
            let desc = vec![*id as f32; 32];

            // ADD PIXEL NOISE
            let pixel_dist = Normal::new(0.0, 5.0)?; // 0.5 pixel std dev
            let noisy_uv = uv + Vector2::new(pixel_dist.sample(rng), pixel_dist.sample(rng));

            let landmark_id = match map.find_match(&desc, threshold) {
                Some(m_id) => m_id,
                None => {
                    // Instead of global noise, add small noise to the LOCAL coordinates
                    let local_pt = camera_isometry.inverse() * true_landmark_position;
                    let noisy_local = local_pt
                        + Vector3::new(
                            rng.random_range(-0.02..0.02),
                            rng.random_range(-0.02..0.02),
                            rng.random_range(-0.02..0.02),
                        );

                    // Transform back to global for the backend
                    let noisy_global = camera_isometry * noisy_local;

                    map.add_landmark(noisy_global, desc);
                    backend.add_landmark_variable(*id, noisy_global);
                    backend.add_weak_landmark_prior(*id, noisy_global, 0.001);
                    *id
                }
            };

            backend.add_projection(pose_id, landmark_id, noisy_uv, camera.clone(), px_weight);
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
            // Only exports landmarks that were successfully added to the solver
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

fn apply_noise_to_isometry(
    isometry: Isometry3<f64>,
    translation_std: f64,
    rotation_std: f64,
    rng: &mut impl Rng,
) -> Isometry3<f64> {
    let translation = Vector3::new(
        rng.random_range(-translation_std..translation_std),
        rng.random_range(-translation_std..translation_std),
        rng.random_range(-translation_std..translation_std),
    );
    let rotation = nalgebra::UnitQuaternion::from_euler_angles(
        rng.random_range(-rotation_std..rotation_std),
        rng.random_range(-rotation_std..rotation_std),
        rng.random_range(-rotation_std..rotation_std),
    );
    isometry * Isometry3::from_parts(translation.into(), rotation)
}

fn apply_gaussian_noise_to_isometry(
    isometry: Isometry3<f64>,
    translation_distribution: &Normal<f64>,
    rotation_distribution: &Normal<f64>,
    rng: &mut impl Rng,
) -> Isometry3<f64> {
    let translation = Vector3::new(
        translation_distribution.sample(rng),
        translation_distribution.sample(rng),
        translation_distribution.sample(rng),
    );
    let rotation = nalgebra::UnitQuaternion::from_euler_angles(
        rotation_distribution.sample(rng),
        rotation_distribution.sample(rng),
        rotation_distribution.sample(rng),
    );
    isometry * Isometry3::from_parts(translation.into(), rotation)
}
