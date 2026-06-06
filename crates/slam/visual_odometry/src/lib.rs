mod feature_extractor;
mod features;
mod matcher_3d;
mod pipeline;

pub use pipeline::{Error, Parameters, Pipeline};

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use ndarray::Array2;

    use crate::{Parameters, Pipeline};

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
        let mut pipeline = Pipeline::new(Parameters {
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
