use apex_solver::{
    CameraModel, LieGroup, PinholeCamera, SE3,
    camera_models::{DistortionModel, PinholeParams},
    core::problem::VariableEnum,
};
use color_eyre::Result;
use nalgebra::Point3;
use nalgebra::{Isometry3, Vector3};
use rand::Rng;
use slam::backend::smart_matcher::SmartMatcher;
use slam::backend::{map::LandmarkMap, slam_backend::Backend};

fn main() -> Result<()> {
    color_eyre::install()?;

    let mut backend = Backend::new();
    let mut map = LandmarkMap::new();

    let mut rng = rand::rng();

    // 1. Generate a 3D Cloud of 50 Landmarks
    let mut true_landmarks = Vec::new();
    for _ in 0..50 {
        true_landmarks.push(Point3::new(
            rng.random_range(-5.0..5.0),
            rng.random_range(-5.0..5.0),
            rng.random_range(5.0..15.0), // Keep them in front of the camera
        ));
    }

    // Define a simple Pinhole Camera (fx, fy, cx, cy)
    let pinhole = PinholeParams::new(500.0, 500.0, 320.0, 240.0)?;
    let distortion = DistortionModel::None;
    let camera = PinholeCamera::new(pinhole, distortion)?;

    let mut matcher = SmartMatcher::new(10);

    // 1. Setup: Create some "Real World" landmarks to observe
    let world_points = vec![Point3::new(1.0, 1.0, 5.0), Point3::new(-1.0, 0.5, 8.0)];

    for i in 0..50 {
        // True path is a slight curve
        let true_iso = Isometry3::translation(i as f64 * 0.1, (i as f64 * 0.05).sin(), 0.0);

        // Add "Odometry Noise" to the initial guess we give the backend
        let noisy_iso = Isometry3::translation(
            true_iso.translation.x + rng.random_range(-0.02..0.02),
            true_iso.translation.y + rng.random_range(-0.02..0.02),
            0.0,
        );
        let pose_id = backend.add_pose(SE3::from_isometry(noisy_iso));

        let mut frame_lms = Vec::new();

        for (idx, &true_pt) in true_landmarks.iter().enumerate() {
            let local_pt = true_iso.inverse() * true_pt;

            // Only observe if it's within the camera frustum
            if local_pt.z > 0.5 && local_pt.z < 20.0 {
                // Add "Pixel Noise" to the measurement
                let mut uv = camera.project(&local_pt.coords)?;
                uv.x += rng.random_range(-0.5..0.5);
                uv.y += rng.random_range(-0.5..0.5);

                let mock_desc = vec![idx as f32; 32];
                let matched_id = matcher.find_match(&map, &mock_desc, false);

                let lm_id = match matched_id {
                    Some(id) => id,
                    None => {
                        // Initialize with a noisy position
                        let noisy_pt = true_pt
                            + Vector3::new(
                                rng.random_range(-0.1..0.1),
                                rng.random_range(-0.1..0.1),
                                rng.random_range(-0.1..0.1),
                            );
                        let id = map.add_landmark(noisy_pt, mock_desc);
                        backend.add_landmark_variable(id, noisy_pt);
                        id
                    }
                };

                backend.add_projection(pose_id, lm_id, uv, camera.clone());
                frame_lms.push(lm_id);
            }
        }
        matcher.record_frame(pose_id, frame_lms);
    }

    // 3. Solve
    let result = backend.optimize();
    // println!(
    //     "Optimization finished. result parameters: {}",
    //     result.parameters.values()
    // );
    dbg!(result.parameters.clone());
    dbg!(result.status);
    println!("Initial Cost: {}", result.initial_cost);
    println!("Final Cost:   {}", result.final_cost);

    // 2. Calculate Root Mean Squared Error (RMSE)
    // The 'cost' in Levenberg-Marquardt is usually 0.5 * sum(residuals^2)
    let num_residuals = result.parameters.len() as f64; // Approximation
    let rmse = (2.0 * result.final_cost / num_residuals).sqrt();

    println!("Estimated RMSE: {:.4} units/pixels", rmse);

    // 3. Verify specific Landmark positions
    if let Some(var_enum) = result.parameters.get("l0") {
        if let VariableEnum::Rn(rn_var) = var_enum {
            // Access the underlying DVector/data from the Variable wrapper
            let data = rn_var.value.data();
            println!(
                "Optimized Landmark 0: [{:.2}, {:.2}, {:.2}]",
                data[0], data[1], data[2]
            );
        }
    }

    Ok(())
}
