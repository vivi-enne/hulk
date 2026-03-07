use std::path::PathBuf;

use linear_algebra::nalgebra::Isometry3;
use ndarray::{Array2, ArrayView2};
use thiserror::Error;

use crate::{
    feature_extractor::{XFeatError, XFeatModel},
    matcher_3d_python::{Matcher3DError, Matcher3DPython},
};

#[derive(Debug, Error)]
pub enum VisualOdometryError {
    #[error("feature extraction failed")]
    FeatureExtraction(#[from] XFeatError),
    #[error("matcher failed")]
    Matcher(#[from] Matcher3DError),
}

pub struct VisualOdometryPipeline {
    extractor: XFeatModel,
    matcher: Matcher3DPython,
}

pub struct VisualOdometryParameters {
    pub xfeat_model_path: PathBuf,
    pub left_calibration: Array2<f32>,
    pub right_calibration: Array2<f32>,
}

impl VisualOdometryPipeline {
    pub fn new(params: VisualOdometryParameters) -> Result<Self, VisualOdometryError> {
        let extractor = XFeatModel::new(&params.xfeat_model_path)?;
        let matcher =
            Matcher3DPython::initialize(params.left_calibration, params.right_calibration)?;
        Ok(Self { extractor, matcher })
    }

    pub fn step(
        &mut self,
        left_image: ArrayView2<u8>,
        right_image: ArrayView2<u8>,
    ) -> Result<Isometry3<f32>, VisualOdometryError> {
        let features = self.extractor.extract(left_image, right_image)?;
        let odometry = self.matcher.step(features)?;
        Ok(odometry.isometry.isometry)
    }
}
