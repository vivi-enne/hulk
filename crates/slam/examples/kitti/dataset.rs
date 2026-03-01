use std::path::PathBuf;

pub struct KittiOdometryCalibration {
    p0: Vec<f32>,
    p1: Vec<f32>,
    p2: Vec<f32>,
    p3: Vec<f32>,
}

pub struct KittiOdometryItem {
    pub time: f32,
    pub left_image: ndarray::Array3<f32>,
    pub right_image: ndarray::Array3<f32>,
}

pub struct KittiOdometrySequence {
    index: usize,
    path: PathBuf,
    times: Vec<usize>,
    calibration: KittiOdometryCalibration,
}

impl Iterator for &mut KittiOdometrySequence {
    type Item;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

impl KittiOdometryItem {}
