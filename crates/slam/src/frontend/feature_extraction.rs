pub mod xfeat;

use color_eyre::Result;
use nalgebra::Point2;

pub trait FeatureExtractor {
    type FeatureSet: FeatureDescriptorSet;

    fn extract(&mut self, images: GrayscaleNCHW) -> Result<Vec<Self::FeatureSet>>;
}

pub trait FeatureDescriptorSet {
    fn len(&self) -> usize;

    fn position(&self) -> Point2<f32>;
    fn match_keypoints(&self, other: &Self) -> f64;
}

pub struct GrayscaleNCHW {
    n: usize,
    height: usize,
    width: usize,
    data: Vec<f32>,
}
