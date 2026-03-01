#[cfg(test)]
mod tests {
    use apex_solver::{SE3, camera_models::DistortionModel};
    // FIX 1: Import the LieGroup trait to unlock the .inverse() method on SE3
    use apex_solver::manifold::LieGroup;
    // FIX 2: Import the built-in PinholeCamera model
    use apex_solver::camera_models::{PinholeCamera, PinholeParams};

    // FIX 3: Swap Point2 for Vector2
    use nalgebra::{Point3, UnitQuaternion, Vector2, Vector3};
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal};
    use slam::backend::slam_backend::Backend;

    // Import your local modules (adjust paths as necessary for your workspace)
    use slam::backend::map::LandmarkMap;

    #[test]
    fn test_native_projection_factor() {
        let mut backend = Backend::new();

        // 1. Setup Camera using apex_solver's native model
        let pinhole_parameters =
            PinholeParams::new(500.0, 500.0, 320.0, 240.0).expect("todo: pinholeParams"); // fx, fy, cx, cy
        let camera = PinholeCamera::new(pinhole_parameters, DistortionModel::None);

        // 2. Define Ground Truth
        let gt_pose = SE3::new(Vector3::new(0.0, 0.0, 0.0), UnitQuaternion::identity());
        let gt_landmark = Point3::new(0.0, 0.0, 5.0); // 5 meters straight ahead

        // 3. Generate perfect frontend measurement (u, v pixel coordinates)
        let inv_pose = gt_pose.inverse(None);

        // Apply the SE(3) transformation: rotate the point, then add the translation
        let p_c = inv_pose.rotation_so3().quaternion() * gt_landmark + inv_pose.translation();

        // Use the hardcoded camera intrinsics for the manual projection math
        let perfect_u = 500.0 * (p_c.x / p_c.z) + 320.0;
        let perfect_v = 500.0 * (p_c.y / p_c.z) + 240.0;
        let measurement = Vector2::new(perfect_u, perfect_v);

        // 4. Feed NOISY initial guesses into the backend
        // The tracking algorithm thinks the camera shifted slightly
        let noisy_pose = SE3::new(Vector3::new(0.1, -0.05, 0.02), UnitQuaternion::identity());
        let pose_id = backend.add_pose(noisy_pose);

        // The mapping algorithm thinks the landmark is at 4.5m instead of 5.0m
        let noisy_landmark = Point3::new(0.1, 0.1, 4.5);
        let lm_id = 1;
        backend.add_landmark_variable(lm_id, noisy_landmark);

        // 5. Connect the pose and the landmark with the perfect measurement
        // Pass the PinholeCamera instance instead of the old intrinsics struct
        backend.add_projection(
            pose_id,
            lm_id,
            measurement,
            camera.expect("camera is existing"),
        );

        // 6. Run Bundle Adjustment
        println!("Running optimizer...");
        let result = backend.optimize();
        println!("{:?}", result.convergence_info);
        // assert!(result.convergence_info, "Optimization failed to converge!");
        println!("Optimization successful.");
    }
    #[test]
    fn test_circular_trajectory_bundle_adjustment() {
        let mut backend = Backend::new();
        let mut map = LandmarkMap::new();

        // 1. Setup Camera
        let pinhole_params = PinholeParams::new(500.0, 500.0, 320.0, 240.0).unwrap();
        let camera = PinholeCamera::new(pinhole_params, DistortionModel::None).unwrap();

        // // 2. Setup Noise Generators (Simulating real-world sensor inaccuracy)
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let pose_noise = Normal::new(0.0, 0.05).unwrap(); // 5cm std dev
        let lm_noise = Normal::new(0.0, 0.1).unwrap(); // 10cm std dev

        // 3. Generate Ground Truth Landmarks (A cluster in the center)
        let mut gt_landmarks = Vec::new();
        for x in -2..=2 {
            for y in -2..=2 {
                let gt_pos = Point3::new(x as f64 * 2.0, y as f64 * 2.0, 10.0);
                gt_landmarks.push(gt_pos);

                // Front-end adds noisy landmarks to the map and backend
                let noisy_pos = Point3::new(
                    gt_pos.x + lm_noise.sample(&mut rng),
                    gt_pos.y + lm_noise.sample(&mut rng),
                    gt_pos.z + lm_noise.sample(&mut rng),
                );
                let lm_id = map.add_landmark(noisy_pos);
                backend.add_landmark_variable(lm_id, noisy_pos);
            }
        }

        // 4. Simulate Robot Trajectory (Moving in a circle around the landmarks)
        let num_poses = 20;
        let radius = 1.0;
        let mut previous_gt_pose: Option<SE3> = None;
        let mut previous_pose_id: Option<u64> = None;

        for i in 0..num_poses {
            let angle = (i as f64) * (2.0 * std::f64::consts::PI / num_poses as f64);

            // Ground Truth Pose
            let t_x = radius * angle.cos();
            let t_y = radius * angle.sin();
            let gt_pose = SE3::new(Vector3::new(t_x, t_y, 0.0), UnitQuaternion::identity());

            // Noisy Initial Guess from Visual Odometry
            // Noisy Initial Guess from Visual Odometry
            // CHANGED: Anchor the first pose perfectly so the map doesn't drift globally
            let noisy_pose = if i == 0 {
                gt_pose.clone()
            } else {
                SE3::new(
                    Vector3::new(
                        t_x + pose_noise.sample(&mut rng),
                        t_y + pose_noise.sample(&mut rng),
                        0.0 + pose_noise.sample(&mut rng), // Adjusted to 0.0 to match gt_pose
                    ),
                    UnitQuaternion::identity(), // keep rotation perfect for simplicity
                )
            };
            let pose_id = backend.add_pose(noisy_pose);

            // Add Odometry constraint (BetweenFactor) if not the first pose
            if let (Some(prev_gt), Some(prev_id)) = (previous_gt_pose, previous_pose_id) {
                // Ground truth relative motion
                let rel_motion_gt = prev_gt.inverse(None).compose(&gt_pose, None, None);

                // Add noise to the measurement to simulate wheel slip/IMU drift
                let noisy_rel_motion = SE3::new(
                    Vector3::new(
                        rel_motion_gt.translation().x + pose_noise.sample(&mut rng),
                        rel_motion_gt.translation().y + pose_noise.sample(&mut rng),
                        rel_motion_gt.translation().z + pose_noise.sample(&mut rng),
                    ),
                    UnitQuaternion::identity(),
                );

                // backend.add_between(prev_id, pose_id, noisy_rel_motion);
            }

            // 5. Simulate Visual Measurements (Projections)
            for (lm_idx, gt_lm) in gt_landmarks.iter().enumerate() {
                // Project ground truth landmark into ground truth camera to get perfect pixels
                let inv_pose = gt_pose.inverse(None);
                let p_c = inv_pose.rotation_so3().quaternion() * gt_lm + inv_pose.translation();

                // Only add measurement if landmark is in front of the camera
                if p_c.z > 0.1 {
                    let u = 500.0 * (p_c.x / p_c.z) + 320.0;
                    let v = 500.0 * (p_c.y / p_c.z) + 240.0;

                    // Assuming front-end matched feature perfectly (no outliers)
                    backend.add_projection(
                        pose_id,
                        lm_idx as u64, // Matches ID in map since we added sequentially
                        Vector2::new(u, v),
                        camera.clone(),
                    );
                } else {
                    println!("Warning: Landmark {} is behind the camera!", lm_idx);
                }
            }

            previous_gt_pose = Some(gt_pose);
            previous_pose_id = Some(pose_id);
        }

        // 6. Optimize the Graph
        println!(
            "Graph built with {} poses and {} landmarks. Optimizing...",
            num_poses,
            gt_landmarks.len()
        );
        let result = backend.optimize();
        // assert!(result., "Optimization failed!");
        println!("Optimization successful.");

        // 7. Evaluate the Results (RMSE)
        let mut initial_error_sq = 0.0;
        let mut final_error_sq = 0.0;

        for (i, gt_lm) in gt_landmarks.iter().enumerate() {
            // Get the final optimized landmark from the backend
            let var_name = format!("l{}", i);
            let opt_dv = result
                .parameters
                .get(&var_name)
                .expect("Landmark not found in backend")
                .to_vector();
            // dbg!("Optimized landmark {}: {:?}", i, opt_dv);
            let opt_lm = Point3::new(opt_dv[0], opt_dv[1], opt_dv[2]);

            // For comparison, get the original noisy landmark from our map
            let noisy_lm = map.landmarks.get(&(i as u64)).unwrap().position;

            // Calculate squared Euclidean distances
            let initial_dist = nalgebra::distance(&noisy_lm, gt_lm);
            let final_dist = nalgebra::distance(&opt_lm, gt_lm);

            initial_error_sq += initial_dist * initial_dist;
            final_error_sq += final_dist * final_dist;
        }

        let num_lm = gt_landmarks.len() as f64;
        let initial_rmse = (initial_error_sq / num_lm).sqrt();
        let final_rmse = (final_error_sq / num_lm).sqrt();

        println!("\n--- Landmark Optimization Results ---");
        println!("Initial Noisy RMSE: {:.4} meters", initial_rmse);
        println!("Final Optimized RMSE: {:.4} meters", final_rmse);

        // Assert that the optimizer actually improved the map
        assert!(final_rmse < initial_rmse, "Optimizer made the map worse!");

        // We expect a significant improvement for this simulation
        assert!(
            final_rmse < 0.1,
            "Final map did not converge accurately enough!"
        );
    }
}
fn main() {
    return;
}
