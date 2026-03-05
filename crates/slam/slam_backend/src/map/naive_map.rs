use std::mem::transmute;

use coordinate_systems::SlamMap;
use linear_algebra::Point3;
use ndarray::ArrayView2;

pub type Descriptor = [f32; 32];

pub struct LandmarkMap {
    keypoints: Vec<Point3<SlamMap>>,
    descriptors: Vec<Descriptor>,
}

impl LandmarkMap {
    fn keypoints(&self) -> &[Point3<SlamMap>] {
        &self.keypoints
    }

    fn keypoint_array(&self) -> ArrayView2<'_, f32> {
        const { assert!(size_of::<Point3<SlamMap>>() == 3 * size_of::<f32>()) }
        let data: &[f32] = unsafe { transmute(self.keypoints.as_slice()) };
        ArrayView2::from_shape((self.keypoints.len(), 3), data)
            .expect("failed to convert points to array view")
    }

    fn descriptors(&self) -> &[Descriptor] {
        &self.descriptors
    }

    fn descriptor_array(&self) -> ArrayView2<'_, f32> {
        const { assert!(size_of::<Descriptor>() == 32 * size_of::<f32>()) }
        let data: &[f32] = unsafe { transmute(self.keypoints.as_slice()) };
        ArrayView2::from_shape((self.keypoints.len(), 3), data)
            .expect("failed to convert points to array view")
    }
}
