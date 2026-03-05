use apex_solver::{
    CameraModel, PinholeCamera,
    camera_models::{DistortionModel, PinholeParams},
    manifold::se3::SE3,
};
use color_eyre::Result;
use nalgebra::{Isometry3, Point3, Vector2, Vector3};
use std::collections::{HashMap, HashSet};

use crate::backend::{map::LandmarkMap, slam_backend::Backend};

// Assuming the rest of your Backend struct and LandmarkMap are in this file...

/// 1. The missing SmartMatcher definition
pub struct SmartMatcher {
    pub observation_history: Vec<(u64, Vec<u64>)>, // (pose_id, landmark_ids)
    pub window_size: usize,
}

impl SmartMatcher {
    pub fn new(window_size: usize) -> Self {
        Self {
            observation_history: Vec::new(),
            window_size,
        }
    }

    pub fn find_match(
        &self,
        map: &LandmarkMap,
        query_descriptor: &[f32],
        is_loop_closure: bool,
    ) -> Option<u64> {
        let threshold = 0.5;

        if is_loop_closure {
            // Global search
            map.find_match(query_descriptor, threshold)
        } else {
            // Local window search
            let recent_landmarks: HashSet<u64> = self
                .observation_history
                .iter()
                .rev()
                .take(self.window_size)
                .flat_map(|(_, ids)| ids.iter().cloned())
                .collect();

            recent_landmarks.into_iter().find(|&id| {
                if let Some(lm) = map.landmarks.get(&id) {
                    compute_distance(&lm.descriptor, query_descriptor) < threshold
                } else {
                    false
                }
            })
        }
    }

    pub fn record_frame(&mut self, pose_id: u64, seen_landmarks: Vec<u64>) {
        self.observation_history.push((pose_id, seen_landmarks));
        if self.observation_history.len() > 1000 {
            self.observation_history.remove(0);
        }
    }
}

/// 2. Helper for descriptor distance
fn compute_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}
