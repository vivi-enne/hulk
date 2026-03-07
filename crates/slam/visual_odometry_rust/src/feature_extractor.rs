use std::path::Path;

use ndarray::{Array4, ArrayView2, Ix2, Ix3, Zip, s};
use ort::{
    execution_providers::{
        ROCmExecutionProvider, TensorRTExecutionProvider, WebGPUExecutionProvider,
    },
    inputs,
    session::{Session, builder::GraphOptimizationLevel},
    sys::OrtApi,
    value::TensorRef,
};
use thiserror::Error;

use crate::interface::XFeatOutput;

#[derive(Debug, Error)]
pub enum XFeatError {
    #[error("model not loaded")]
    OrtError(#[from] ort::Error),
    #[error(
        "left and right image dimensions mismatch ({left_width}x{left_height} vs {right_width}x{right_height})"
    )]
    ImageDimensionsMismatch {
        left_width: usize,
        left_height: usize,
        right_width: usize,
        right_height: usize,
    },
    #[error(
        "new image dimensions are different from stored dimensions ({stored_width}x{stored_height} vs {new_width}x{new_height})"
    )]
    NewImageDimensionsMismatch {
        stored_width: usize,
        stored_height: usize,
        new_width: usize,
        new_height: usize,
    },
    #[error("unexpected dimensionality")]
    UnexpectedDimensionality(#[from] ndarray::ShapeError),
}

pub struct XFeatModel {
    model: Session,
    storage: Option<Array4<f32>>,
}

impl XFeatModel {
    pub fn new(model_path: impl AsRef<Path>) -> Result<Self, XFeatError> {
        let model = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers([TensorRTExecutionProvider::default().build()])?
            .commit_from_file(model_path)?;
        Ok(Self {
            model,
            storage: None,
        })
    }

    pub fn extract(
        &mut self,
        left_image: ArrayView2<u8>,
        right_image: ArrayView2<u8>,
    ) -> Result<XFeatOutput, XFeatError> {
        self.load_images_into_storage(left_image, right_image)?;
        let tensor = TensorRef::from_array_view(
            self.storage.as_ref().expect("failed to get storage tensor"),
        )?;

        let outputs = self.model.run(inputs! {
            "input" => tensor,
        })?;

        let keypoints = outputs["keypoints"]
            .try_extract_array::<i64>()?
            .into_dimensionality::<Ix3>()?;
        let scores = outputs["scores"]
            .try_extract_array::<f32>()?
            .into_dimensionality::<Ix2>()?;
        let descriptors = outputs["descriptors"]
            .try_extract_array::<f32>()?
            .into_dimensionality::<Ix3>()?;

        Ok(XFeatOutput::new(keypoints, scores, descriptors))
    }

    fn load_images_into_storage(
        &mut self,
        left_image: ArrayView2<u8>,
        right_image: ArrayView2<u8>,
    ) -> Result<(), XFeatError> {
        let (left_height, left_width) = left_image.dim();
        let (right_height, right_width) = right_image.dim();
        if left_width != right_width || left_height != right_height {
            return Err(XFeatError::ImageDimensionsMismatch {
                left_width,
                left_height,
                right_width,
                right_height,
            });
        }

        let cropped_height = left_height / 32 * 32;
        let cropped_width = left_width / 32 * 32;

        let storage = self
            .storage
            .get_or_insert_with(|| Array4::default((2, 1, cropped_height, cropped_width)));

        let (_, _, storage_height, storage_width) = storage.dim();
        if storage_height != cropped_height || storage_width != cropped_width {
            return Err(XFeatError::NewImageDimensionsMismatch {
                stored_width: storage_width,
                stored_height: storage_height,
                new_width: cropped_width,
                new_height: cropped_height,
            });
        }

        Zip::from(&mut storage.slice_mut(s![0, 0, .., ..]))
            .and(left_image.slice(s![..cropped_height, ..cropped_width]))
            .for_each(|target: &mut f32, &source: &u8| {
                *target = source as f32 / 255.0;
            });
        Zip::from(&mut storage.slice_mut(s![1, 0, .., ..]))
            .and(right_image.slice(s![..cropped_height, ..cropped_width]))
            .for_each(|target: &mut f32, &source: &u8| {
                *target = source as f32 / 255.0;
            });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use crate::feature_extractor::XFeatModel;

    #[test]
    fn infer_images() {
        let mut model = XFeatModel::new("xfeat.onnx").unwrap();
        let left_image = Array2::zeros([480, 640]);
        let right_image = Array2::zeros([480, 640]);

        let outputs = model
            .extract(left_image.view(), right_image.view())
            .expect("failed to infer");
        dbg!(outputs);
    }
}
