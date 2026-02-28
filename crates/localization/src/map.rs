// store simulated landmarks
use nalgebra::Point3;
use std::collections::HashMap;

pub type LandmarkId = u64;

#[derive(Clone)]
pub struct Landmark {
    pub id: LandmarkId,
    pub position: Point3<f64>,
}

pub struct LandmarkMap {
    pub landmarks: HashMap<LandmarkId, Landmark>,
    next_id: LandmarkId,
}

impl LandmarkMap {
    pub fn new() -> Self {
        Self { landmarks: HashMap::new(), next_id: 0 }
    }

    pub fn add_landmark(&mut self, position: Point3<f64>) -> LandmarkId {
        let id = self.next_id;
        self.next_id += 1;
        let lm = Landmark { id, position };
        self.landmarks.insert(id, lm);
        id
    }
}
