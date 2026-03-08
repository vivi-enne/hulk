use std::{
    collections::{BTreeMap, HashMap},
    fs::File,
    io::BufWriter,
    time::Instant,
};

use apex_solver::{
    CameraModel, LieGroup, PinholeCamera, SE3,
    camera_models::{DistortionModel, PinholeParams},
    core::problem::VariableEnum,
    optimizer::SolverResult,
};
use color_eyre::Result;
use nalgebra::{Isometry3, Point3, Vector2, Vector3};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};
use serde::Serialize;
use slam_backend::backend::{map::LandmarkMap, slam_backend::Backend};
#[derive(Serialize, Clone, Debug)]
struct PoseStep {
    id: u64,
    true_pose: (f64, f64, f64),
    optimized_pose: (f64, f64, f64),
}

#[derive(Serialize, Clone, Debug)]
struct LandmarkStep {
    lm_id: u64,
    true_pos: [f64; 3],
    opt_pos: [f64; 3],
}

// New struct to capture the graph state at every step
#[derive(Serialize, Clone, Debug)]
struct FrameData {
    step: u64,
    poses: Vec<PoseStep>,
    landmarks: Vec<LandmarkStep>,
}

fn main() -> Result<()> {
    let mut backend = Backend::new();
    let mut map = LandmarkMap::new();

    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()
        .ok();

    let pinhole = PinholeParams::new(500.0, 500.0, 320.0, 240.0)?;
    let camera = PinholeCamera::new(pinhole, DistortionModel::None)?;

    let mut true_poses = BTreeMap::new();
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let threshold = 0.1;

    let true_landmarks = generate_true_landmarks(150, &mut rng);
    let mut last_seen_landmarks: HashMap<u64, u64> = HashMap::new();

    // Initial State
    let true_iso = Isometry3::translation(5.0, 0.0, 0.0);
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
        &mut last_seen_landmarks,
        threshold,
        &mut rng,
    )?;

    // Store the snapshots
    let mut frames: Vec<FrameData> = Vec::new();
    let mut before_optimization_time = Instant::now();

    // Optimize step 0
    let result = backend.optimize()?;
    println!("{}", before_optimization_time.elapsed().as_secs_f64());

    backend.update_initial_values(&result);

    frames.push(extract_frame(0, &result, &true_poses, &true_landmarks));

    let mut is_loop_closure;

    let translation_std = 0.1;
    let rotation_std = 0.001;
    let laps = 6;
    let steps_per_lap = 100;
    let total_steps = laps * steps_per_lap;
    let window_size = 50; // only optimize the last 15 poses

    for i in 1..=total_steps {
        println!("Processing step {}/{}...", i, total_steps);

        // // let t = i as f64 * 0.2;
        // // let x = t.cos() * 2.0;
        // // let y = t.sin() * 2.0;
        // // let z = t * 0.3; // Upward movement

        // // Circular Room Trajectory
        // let t = (i as f64) * (2.0 * std::f64::consts::PI / (total_steps - 20) as f64);
        // let x = t.cos() * 5.0;
        // let y = t.sin() * 5.0;
        // let z = 0.0;

        let t = (i as f64) * (laps as f64 * 2.0 * std::f64::consts::PI / total_steps as f64);

        // Figure-8 (Lissajous) Parametric Equations (Bounded to roughly 12x8m)
        let x = 6.0 * t.sin();
        let y = 4.0 * (2.0 * t).sin();
        let z = t * 0.1;

        // Calculate heading (yaw) using the derivatives dx/dt and dy/dt
        let dx = 6.0 * t.cos();
        let dy = 8.0 * (2.0 * t).cos();
        let yaw = dy.atan2(dx);

        // Camera looks along the circle path
        let rotation =
            nalgebra::UnitQuaternion::from_euler_angles(0.0, t + std::f64::consts::FRAC_PI_2, yaw);
        let true_iso = Isometry3::from_parts(nalgebra::Translation3::new(x, y, z), rotation);
        let true_t_cw = true_iso.inverse();

        let true_relative_delta = prev_true_t_cw.inverse() * true_t_cw.clone();

        let trans_dist = Normal::new(0.0, translation_std)?;
        let rot_dist = Normal::new(0.0, rotation_std)?;

        let noisy_delta =
            apply_gaussian_noise_to_isometry(true_relative_delta, &trans_dist, &rot_dist, &mut rng);
        let odom_weight = 1.0 / (translation_std * translation_std);

        current_guess_t_cw = prev_guess_t_cw * noisy_delta.clone();
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
            &mut last_seen_landmarks,
            threshold,
            &mut rng,
        )?;

        is_loop_closure = i % steps_per_lap == 0;
        if is_loop_closure {
            println!("Lap complete! Running Global BA...");
            backend.unfix_all_variables();

            // Re-anchor the origin so the map doesn't float away!
            for idx in 0..6 {
                backend.problem.fix_variable("x0", idx);
            }

            before_optimization_time = Instant::now();
            let optimizer_result = backend.optimize();
            println!(
                "Global BA Time: {} seconds",
                before_optimization_time.elapsed().as_secs_f64()
            );

            match optimizer_result {
                Ok(result) => {
                    backend.update_initial_values(&result);
                    frames.push(extract_frame(
                        i as u64,
                        &result,
                        &true_poses,
                        &true_landmarks,
                    ));
                }
                Err(e) => println!("Global BA Failed: {}. Continuing with Dead-Reckoning.", e),
            }
        } else {
            // Normal tracking step
            before_optimization_time = Instant::now();
            let optimizer_result =
                backend.optimize_sliding_window(window_size, &last_seen_landmarks);
            println!(
                "Sliding Window Optimization Time: {} seconds",
                before_optimization_time.elapsed().as_secs_f64()
            );

            match optimizer_result {
                Ok(result) => {
                    backend.update_initial_values(&result);
                    frames.push(extract_frame(
                        i as u64,
                        &result,
                        &true_poses,
                        &true_landmarks,
                    ));
                }
                Err(e) => {
                    // just log a warning and let the robot coast on odometry
                    println!("Local tracking wobble at step {}: {}. Coasting...", i, e);
                }
            }
        }
    }

    let file = File::create("apex_pose_graph.json")?;
    serde_json::to_writer(BufWriter::new(file), &frames)?;
    Ok(())
}

fn extract_frame(
    step: u64,
    result: &SolverResult<HashMap<String, VariableEnum>>,
    true_poses: &BTreeMap<u64, Point3<f64>>,
    true_landmarks: &BTreeMap<u64, Point3<f64>>,
) -> FrameData {
    let mut poses = Vec::new();
    let mut landmarks = Vec::new();

    for (name, var_enum) in &result.parameters {
        if name.starts_with('x') {
            let id: u64 = name[1..].parse().unwrap();
            if let (VariableEnum::SE3(se3), Some(tp)) = (var_enum, true_poses.get(&id)) {
                let op = se3.value.inverse(None).translation();
                poses.push(PoseStep {
                    id,
                    true_pose: (tp.x, tp.y, tp.z),
                    optimized_pose: (op.x, op.y, op.z),
                });
            }
        } else if name.starts_with('l') {
            let id: u64 = name[1..].parse().unwrap();
            if let (VariableEnum::Rn(rn), Some(tp)) = (var_enum, true_landmarks.get(&id)) {
                let op = rn.value.data();
                landmarks.push(LandmarkStep {
                    lm_id: id,
                    true_pos: [tp.x, tp.y, tp.z],
                    opt_pos: [op[0], op[1], op[2]],
                });
            }
        }
    }

    poses.sort_by_key(|p| p.id);
    landmarks.sort_by_key(|l| l.lm_id);
    FrameData {
        step,
        poses,
        landmarks,
    }
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
    last_seen_landmarks: &mut HashMap<u64, u64>,
    threshold: f32,
    rng: &mut ChaCha8Rng,
) -> Result<()> {
    let mut frame_landmarks = Vec::new();
    // Tie the weight mathematically to the noise variance
    let px_std = 5.0;
    let px_weight = 1.0 / (px_std * px_std); // Weight becomes 0.04

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
                    let noise_range = 0.1; // 2 cm noise
                    let noisy_local = local_pt
                        + Vector3::new(
                            rng.random_range(-noise_range..noise_range),
                            rng.random_range(-noise_range..noise_range),
                            rng.random_range(-noise_range..noise_range),
                        );

                    // Transform back to global for the backend
                    let noisy_global = camera_isometry * noisy_local;

                    map.add_landmark(noisy_global, desc);
                    backend.add_landmark_variable(*id, noisy_global);
                    backend.add_weak_landmark_prior(*id, noisy_global, 0.1);
                    *id
                }
            };

            last_seen_landmarks.insert(landmark_id, pose_id);

            backend.add_projection(pose_id, landmark_id, noisy_uv, camera.clone(), px_weight);
            frame_landmarks.push(landmark_id);
        }
    }

    Ok(())
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
