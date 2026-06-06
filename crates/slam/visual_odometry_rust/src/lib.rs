mod feature_extractor;
mod features;
mod matcher_3d;
mod pipeline;

pub use feature_extractor::{XFeatError, XFeatModel};
pub use features::XFeatOutput;
pub use matcher_3d::{Matcher3D, Matcher3DError, MatcherOutput};
pub use pipeline::{VisualOdometryError, VisualOdometryParameters, VisualOdometryPipeline};

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use ndarray::Array2;

    use crate::{VisualOdometryParameters, VisualOdometryPipeline};

    #[test]
    fn infer_and_match() {
        let left_calibration = Array2::from_shape_vec(
            (3, 4),
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        )
        .unwrap();
        let right_calibration = Array2::from_shape_vec(
            (3, 4),
            vec![1.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        )
        .unwrap();
        let mut pipeline = VisualOdometryPipeline::new(VisualOdometryParameters {
            xfeat_model_path: PathBuf::from("xfeat.onnx"),
            left_calibration,
            right_calibration,
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
