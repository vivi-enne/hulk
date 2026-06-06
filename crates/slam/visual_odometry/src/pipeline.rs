use std::path::PathBuf;

use linear_algebra::nalgebra::Isometry3;
use ndarray::{Array2, ArrayView2};
use thiserror::Error;

use crate::{
    feature_extractor::{XFeatError, XFeatModel},
    matcher_3d::{Matcher3D, Matcher3DError},
};

#[derive(Debug, Error)]
#[error(transparent)]
pub struct Error(#[from] ErrorKind);

#[derive(Debug, Error)]
enum ErrorKind {
    #[error("feature extraction failed")]
    FeatureExtraction(#[source] XFeatError),
    #[error("matcher failed")]
    Matcher(#[source] Matcher3DError),
}

impl From<XFeatError> for Error {
    fn from(error: XFeatError) -> Self {
        Self(ErrorKind::FeatureExtraction(error))
    }
}

impl From<Matcher3DError> for Error {
    fn from(error: Matcher3DError) -> Self {
        Self(ErrorKind::Matcher(error))
    }
}

pub struct Pipeline {
    extractor: XFeatModel,
    matcher: Matcher3D,
}

pub struct Parameters {
    pub xfeat_model_path: PathBuf,
    pub left_calibration: Array2<f32>,
    pub right_calibration: Array2<f32>,
}

impl Pipeline {
    pub fn new(params: Parameters) -> Result<Self, Error> {
        let extractor = XFeatModel::new(&params.xfeat_model_path)?;
        let matcher = Matcher3D::initialize(params.left_calibration, params.right_calibration)?;
        Ok(Self { extractor, matcher })
    }

    pub fn step(
        &mut self,
        left_image: ArrayView2<u8>,
        right_image: ArrayView2<u8>,
    ) -> Result<Isometry3<f32>, Error> {
        let features = self.extractor.extract(left_image, right_image)?;
        self.matcher.step(features).map_err(Into::into)
    }
}
