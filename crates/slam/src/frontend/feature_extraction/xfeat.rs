use std::path::PathBuf;

use color_eyre::Result;
use ndarray::{Array1, Array2, Axis};
use ort::{
    execution_providers::TensorRTExecutionProvider,
    inputs,
    session::{Session, builder::GraphOptimizationLevel},
    value::Tensor,
};

use super::{FeatureDescriptorSet, FeatureExtractor, GrayscaleNCHW};

pub struct XFeatExtractorParameters {
    xfeat_model_path: PathBuf,
    top_k: usize,
    detection_threshold: f64,
}

pub struct XFeatExtractor {
    /// The XFeat model. Expects inputs to be in [B, 1, H, W] format.
    /// H and W need to be multiples of 32.
    /// Internally RGB images are converted to grayscale using the average over all color channels.
    model: Session,
    parameters: XFeatExtractorParameters,
}

impl XFeatExtractor {
    pub fn new(parameters: XFeatExtractorParameters) -> Result<Self> {
        let model = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers([TensorRTExecutionProvider::default().build()])?
            .commit_from_file(&parameters.xfeat_model_path)?;
        Ok(Self { model, parameters })
    }
}

impl FeatureExtractor for XFeatExtractor {
    type FeatureSet = XFeatureSet;

    fn extract(&mut self, images: GrayscaleNCHW) -> Result<Vec<Self::FeatureSet>> {
        let n = images.n;
        let inputs = Tensor::from_array(([n, 1, images.height, images.width], images.data))?;
        let top_k = Tensor::from_array(([1], vec![self.parameters.top_k as i64]))?;
        let threshold =
            Tensor::from_array(([1], vec![self.parameters.detection_threshold as f32]))?;

        let outputs = self.model.run(inputs! {
            "input" => inputs,
            "topk" => top_k,
            "threshold" => threshold
        })?;
        let keypoints = &outputs["keypoints"].try_extract_array::<i64>()?;
        let scores = &outputs["scores"].try_extract_array::<f32>()?;
        let descriptors = &outputs["descriptors"].try_extract_array::<f32>()?;

        Ok(keypoints
            .axis_iter(Axis(0))
            .zip(scores.axis_iter(Axis(0)))
            .zip(descriptors.axis_iter(Axis(0)))
            // Chained iterators group outputs into nested tuples
            .map(|((positions, scores), descriptors)| XFeatureSet {
                positions: positions.into_owned().into_dimensionality().unwrap(),
                scores: scores.into_owned().into_dimensionality().unwrap(),
                descriptors: descriptors.into_owned().into_dimensionality().unwrap(),
            })
            .collect())
    }
}

pub struct XFeatureSet {
    // 2d position of features [N, 2]
    positions: Array2<i64>,
    // score of each feature [N]
    scores: Array1<f32>,
    // descriptor for each feature [N, 64], with euclidean norm of 1 along the last dimension
    descriptors: Array2<f32>,
}

impl FeatureDescriptorSet for XFeatureSet {
    fn len(&self) -> usize {
        self.scores.dim()
    }

    fn position(&self) -> nalgebra::Point2<f32> {
        todo!()
    }

    fn match_keypoints(&self, other: &Self) -> f64 {
        todo!()
    }
}

pub fn prepare_image(image: GrayscaleNCHW) -> GrayscaleNCHW {
    if image.height % 32 == 0 && image.width % 32 == 0 {
        return image;
    }
    todo!("scale or clip image")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detector_on_checkerboard_image() -> Result<()> {
        // Create a checkerboard pattern (64x64, which is a multiple of 32)
        let size = 64;
        let square_size = 8;
        let mut data = Vec::with_capacity(size * size);

        for y in 0..size {
            for x in 0..size {
                let is_black = ((x / square_size) + (y / square_size)) % 2 == 0;
                let pixel = if is_black { 0.0 } else { 1.0 };
                data.push(pixel);
            }
        }

        let checkerboard_image = GrayscaleNCHW {
            n: 1,
            height: size,
            width: size,
            data,
        };

        // Initialize the detector with the model
        let model_path = PathBuf::from("/home/ole/hulk-stuff/hulk/etc/neural_networks/xfeat.onnx");
        let parameters = XFeatExtractorParameters {
            xfeat_model_path: model_path,
            top_k: 100,
            detection_threshold: 0.0,
        };

        let mut detector = XFeatExtractor::new(parameters)?;

        // Run the detector on the checkerboard image
        let results = detector.extract(checkerboard_image)?;

        // Verify we got results
        assert_eq!(results.len(), 1, "Expected one feature set from one image");

        let feature_set = &results[0];

        // The checkerboard should have many detectable features
        println!("Detected {} keypoints", feature_set.len());
        assert!(
            feature_set.len() > 0,
            "Checkerboard pattern should produce detectable features"
        );

        // Verify the structure of the feature set
        assert_eq!(
            feature_set.positions.dim().1,
            2,
            "Each keypoint should have 2 coordinates (x, y)"
        );
        assert_eq!(
            feature_set.scores.dim(),
            feature_set.len(),
            "Should have one score per keypoint"
        );
        assert_eq!(
            feature_set.descriptors.dim().0,
            feature_set.len(),
            "Should have one descriptor per keypoint"
        );
        assert_eq!(
            feature_set.descriptors.dim().1,
            64,
            "Each descriptor should have 64 dimensions"
        );

        Ok(())
    }
}
