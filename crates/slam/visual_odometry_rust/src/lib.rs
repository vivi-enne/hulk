mod feature_extractor;
mod interface;
mod matcher_3d_python;
mod pipeline;

pub use feature_extractor::{XFeatError, XFeatModel};
pub use matcher_3d_python::{Matcher3DError, Matcher3DPython};
pub use pipeline::{VisualOdometryError, VisualOdometryParameters, VisualOdometryPipeline};

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use ndarray::Array2;

    use crate::{VisualOdometryParameters, VisualOdometryPipeline};

    #[test]
    fn infer_and_match() {
        let mut pipeline = VisualOdometryPipeline::new(VisualOdometryParameters {
            xfeat_model_path: PathBuf::from("xfeat.onnx"),
            left_calibration: Array2::zeros([3, 3]),
            right_calibration: Array2::zeros([3, 3]),
        })
        .unwrap();

        let left_image = Array2::zeros([480, 640]);
        let right_image = Array2::zeros([480, 640]);

        let isometry = pipeline
            .step(left_image.view(), right_image.view())
            .unwrap();

        dbg!(isometry);
    }
}
