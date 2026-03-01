use std::{collections::HashMap, fs::File, io::BufWriter};

use apex_solver::{
    CameraModel, PinholeCamera, SE3,
    camera_models::{DistortionModel, PinholeParams},
    core::problem::VariableEnum,
};
use color_eyre::Result;
use nalgebra::{Isometry3, Point3, Vector3};
use rand::Rng;
use serde::Serialize;
use slam::backend::{map::LandmarkMap, slam_backend::Backend, smart_matcher::SmartMatcher};

#[derive(Serialize)]
struct PoseStep {
    true_pose: (f64, f64, f64),
    optimized_pose: (f64, f64, f64),
}

#[derive(Serialize)]
struct OutputData {
    // This provides the "history" key your plotting file expects
    history: Vec<PoseStep>,
}

#[derive(Serialize)]
struct LandmarkStep {
    lm_id: u64,
    true_pos: [f64; 3],
    opt_pos: [f64; 3],
}

fn main() -> Result<()> {
    let mut backend = Backend::new();
    let mut map = LandmarkMap::new();
    let mut matcher = SmartMatcher::new(10);

    // Define a simple Pinhole Camera (fx, fy, cx, cy)
    let pinhole = PinholeParams::new(500.0, 500.0, 320.0, 240.0)?;
    let distortion = DistortionModel::None;
    let camera = PinholeCamera::new(pinhole, distortion)?;

    let mut true_poses = HashMap::new();
    let mut true_lms = HashMap::new();
    let mut rng = rand::rng();

    // 1. Create a "Cloud" of True Landmarks
    for i in 0..100 {
        let pt = Point3::new(
            rng.random_range(-10.0..10.0),
            rng.random_range(-10.0..10.0),
            rng.random_range(0.0..20.0),
        );
        true_lms.insert(i as u64, pt);
    }

    // 2. Spiral Trajectory Simulation
    for i in 0..100 {
        let t = i as f64 * 0.2;
        let x = t.cos() * 2.0;
        let y = t.sin() * 2.0;
        let z = t * 0.5; // Upward movement

        let true_iso = Isometry3::translation(x, y, z);
        let pose_id = backend.add_pose(SE3::from_isometry(true_iso.clone()));
        true_poses.insert(pose_id, Point3::from(true_iso.translation.vector));

        let mut frame_lms = Vec::new();
        for (id, &pt) in &true_lms {
            let local_pt = true_iso.inverse() * pt;
            if local_pt.z > 0.5 && local_pt.z < 15.0 {
                let uv = camera.project(&local_pt.coords)?;
                let desc = vec![*id as f32; 32];

                let lm_id = match matcher.find_match(&map, &desc, false) {
                    Some(m_id) => m_id,
                    None => {
                        let noisy_pt = pt + Vector3::new(rng.random_range(-0.2..0.2), 0.0, 0.0);
                        map.add_landmark(noisy_pt, desc);
                        backend.add_landmark_variable(*id, noisy_pt);
                        *id
                    }
                };
                backend.add_projection(pose_id, lm_id, uv, camera.clone());
                frame_lms.push(lm_id);
            }
        }
        matcher.record_frame(pose_id, frame_lms);
    }

    let result = backend.optimize();

    // 3. Extraction with Landmarks
    let mut history = Vec::new();
    let mut landmarks_out = Vec::new();

    for (name, var_enum) in &result.parameters {
        if name.starts_with('x') {
            let id: u64 = name[1..].parse()?;
            if let (VariableEnum::SE3(se3), Some(tp)) = (var_enum, true_poses.get(&id)) {
                let op = se3.value.translation();
                history.push((
                    id,
                    PoseStep {
                        true_pose: (tp.x, tp.y, tp.z),
                        optimized_pose: (op.x, op.y, op.z),
                    },
                ));
            }
        } else if name.starts_with('l') {
            let id: u64 = name[1..].parse()?;
            if let (VariableEnum::Rn(rn), Some(tp)) = (var_enum, true_lms.get(&id)) {
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
