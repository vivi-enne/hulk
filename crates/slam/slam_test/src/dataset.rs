use std::path::{Path, PathBuf};

use color_eyre::{Result, eyre::bail};
use image::{ImageBuffer, ImageReader, Rgb};
use ndarray::{Array2, Array3};

pub struct KittiOdometryCalibration {
    pub p0: Array2<f32>,
    pub p1: Array2<f32>,
    pub p2: Array2<f32>,
    pub p3: Array2<f32>,
}

pub struct KittiOdometryItem {
    pub time: f32,
    pub left_image: ndarray::Array3<u8>,
    pub right_image: ndarray::Array3<u8>,
}

pub struct KittiOdometrySequence {
    path: PathBuf,
    times: Vec<f32>,
    pub calibration: KittiOdometryCalibration,
}

impl KittiOdometrySequence {
    pub fn from_path(path: PathBuf) -> Result<Self> {
        let times = std::fs::read_to_string(path.join("times.txt"))?;
        let times = times
            .lines()
            .map(|line| line.parse::<f32>())
            .collect::<Result<Vec<_>, _>>()?;
        let calibration = KittiOdometryCalibration::load(path.join("calib.txt"))?;

        Ok(Self {
            path,
            times,
            calibration,
        })
    }

    pub fn len(&self) -> usize {
        self.times.len()
    }

    pub fn get(&self, index: usize) -> Option<Result<KittiOdometryItem>> {
        let time = self.times.get(index)?;
        let left_image_path = self.path.join("image_0").join(format!("{:06}.png", index));
        let right_image_path = self.path.join("image_1").join(format!("{:06}.png", index));
        Some(KittiOdometryItem::load(
            *time,
            &left_image_path,
            &right_image_path,
        ))
    }
}

impl KittiOdometryItem {
    pub fn load(
        time: f32,
        left_image_path: impl AsRef<Path>,
        right_image_path: impl AsRef<Path>,
    ) -> Result<Self> {
        let left_image = ImageReader::open(left_image_path)?.decode()?.to_rgb8();
        let right_image = ImageReader::open(right_image_path)?.decode()?.to_rgb8();

        Ok(Self {
            time,
            left_image: convert_image(left_image),
            right_image: convert_image(right_image),
        })
    }
}

impl KittiOdometryCalibration {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let mut projections: Vec<_> = content
            .lines()
            .map(|line| {
                line.split_whitespace()
                    .skip(1)
                    .map(|value| value.parse::<f32>())
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect();
        if projections.len() != 4 {
            bail!("expected 4 projections, got {content}")
        }
        let p3 = Array2::from_shape_vec((3, 4), projections.pop().unwrap()?)?;
        let p2 = Array2::from_shape_vec((3, 4), projections.pop().unwrap()?)?;
        let p1 = Array2::from_shape_vec((3, 4), projections.pop().unwrap()?)?;
        let p0 = Array2::from_shape_vec((3, 4), projections.pop().unwrap()?)?;

        Ok(Self { p0, p1, p2, p3 })
    }
}

fn convert_image(image: ImageBuffer<Rgb<u8>, Vec<u8>>) -> Array3<u8> {
    let (width, height) = image.dimensions();
    let raw = image.into_raw();

    Array3::from_shape_vec((height as usize, width as usize, 3), raw).expect("invalid shape")
}
