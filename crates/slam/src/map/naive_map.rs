use coordinate_systems::SlamMap;
use linear_algebra::{Point3, Pose3};

use crate::map::{GlobalLandmarkMap, Landmarks};

pub struct LandmarkMap<Descriptor> {
    keypoints: Vec<Point3<SlamMap>>,
    descriptors: Vec<Descriptor>,
}

impl<Descriptor> Landmarks for LandmarkMap<Descriptor> {
    type Descriptor = Descriptor;

    fn keypoints(&self) -> &[Point3<SlamMap>] {
        &self.keypoints
    }

    fn descriptors(&self) -> &[Self::Descriptor] {
        &self.descriptors
    }
}

impl<Descriptor: Copy> GlobalLandmarkMap for LandmarkMap<Descriptor> {
    type LocalLandmarkMap = LandmarkMap<Descriptor>;

    fn extract(&self, pose: Pose3<SlamMap>, viewport: super::Viewport) -> Self::LocalLandmarkMap {
        struct Aligned;
        let transform = pose.as_transform::<Aligned>().inverse();
        let fovx_tan = (viewport.fovx / 2.).tan();
        let fovy_tan = (viewport.fovy / 2.).tan();

        let (keypoints, descriptors) = self
            .keypoints
            .iter()
            .zip(&self.descriptors)
            .filter(|(keypoint, _)| {
                let keypoint = transform * *keypoint;
                let depth = keypoint.z();
                keypoint.x().abs() <= depth * fovx_tan && keypoint.y().abs() <= depth * fovy_tan
            })
            .map(|(a, b)| (*a, *b))
            .unzip();

        LandmarkMap {
            keypoints,
            descriptors,
        }
    }
}
