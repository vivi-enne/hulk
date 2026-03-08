use nalgebra::Point3;
use std::collections::HashMap;

pub type LandmarkId = u64;

/// Represents a 3D point in the world with visual information
#[derive(Clone)]
pub struct Landmark {
    pub id: LandmarkId,
    pub position: Point3<f64>,
    /// The visual descriptor (e.g., ORB) used for matching.
    /// Using a Vec<f32> as a generic placeholder for descriptors.
    pub descriptor: Vec<f32>,
    /// Track how many times we've seen this to filter noise
    pub observation_count: usize,
}

pub struct LandmarkMap {
    pub landmarks: HashMap<LandmarkId, Landmark>,
    next_id: LandmarkId,
}

impl LandmarkMap {
    pub fn new() -> Self {
        Self {
            landmarks: HashMap::new(),
            next_id: 0,
        }
    }

    /// Try to find an existing landmark that matches a new feature descriptor
    pub fn find_match(&self, feature_descriptor: &[f32], threshold: f32) -> Option<LandmarkId> {
        // In a real system, use a Flann index or KD-Tree for speed.
        // This is a simple brute-force search:
        self.landmarks
            .values()
            .filter(|lm| {
                let distance = compute_distance(&lm.descriptor, feature_descriptor);
                distance < threshold
            })
            .map(|lm| lm.id)
            .next()
    }

    pub fn add_landmark(&mut self, position: Point3<f64>, descriptor: Vec<f32>) -> LandmarkId {
        let id = self.next_id;
        self.next_id += 1;

        let lm = Landmark {
            id,
            position,
            descriptor,
            observation_count: 1,
        };
        self.landmarks.insert(id, lm);
        id
    }
}

/// Helper for descriptor distance (Euclidean example)
fn compute_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}
