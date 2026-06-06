use ndarray::{Array2, Array3, ArrayView2, ArrayView3};

#[derive(Debug, Clone)]
pub struct XFeatOutput {
    pub keypoints: Array3<i64>,
    pub scores: Array2<f32>,
    pub descriptors: Array3<f32>,
}

impl XFeatOutput {
    pub fn new(
        keypoints: ArrayView3<'_, i64>,
        scores: ArrayView2<'_, f32>,
        descriptors: ArrayView3<'_, f32>,
    ) -> Self {
        Self {
            keypoints: keypoints.to_owned(),
            scores: scores.to_owned(),
            descriptors: descriptors.to_owned(),
        }
    }
}
