use coordinate_systems::SlamMap;
use linear_algebra::{Point3, Pose3};

pub mod naive_map;

pub trait GlobalLandmarkMap: Landmarks {
    type LocalLandmarkMap: Landmarks;
    fn extract(&self, pose: Pose3<SlamMap>, viewport: Viewport) -> Self::LocalLandmarkMap;
}

pub trait Landmarks {
    type Descriptor;

    fn keypoints(&self) -> &[Point3<SlamMap>];
    fn descriptors(&self) -> &[Self::Descriptor];
}

pub struct Viewport {
    pub fovx: f32,
    pub fovy: f32,
}
