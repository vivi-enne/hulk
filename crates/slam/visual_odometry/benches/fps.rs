use std::{
    env,
    error::Error,
    hint::black_box,
    io,
    path::{Path, PathBuf},
    time::Duration,
};

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use image::ImageReader;
use ndarray::Array2;
use visual_odometry::{Parameters, Pipeline};

const XFEAT_ENVIRONMENT_VARIABLE: &str = "VISUAL_ODOMETRY_BENCH_XFEAT";
const KITTI_SEQUENCE_ENVIRONMENT_VARIABLE: &str = "VISUAL_ODOMETRY_BENCH_SEQUENCE";
const KITTI_FRAME_LIMIT_ENVIRONMENT_VARIABLE: &str = "VISUAL_ODOMETRY_BENCH_FRAMES";
const DEFAULT_KITTI_FRAME_LIMIT: usize = 64;

struct BenchmarkInput {
    left_calibration: Array2<f32>,
    right_calibration: Array2<f32>,
    frames: Vec<StereoFrame>,
}

struct StereoFrame {
    left_image: Array2<u8>,
    right_image: Array2<u8>,
}

fn benchmark_fps(criterion: &mut Criterion) {
    let model_path = model_path();
    let mut group = criterion.benchmark_group("visual_odometry_fps");
    group.throughput(Throughput::Elements(1));

    group.bench_function("synthetic_480x640", |bencher| {
        let input = synthetic_input(480, 640);
        let mut runner = BenchmarkRunner::new(&model_path, &input);

        bencher.iter(|| black_box(runner.step()));
    });

    if let Some(sequence_path) = env::var_os(KITTI_SEQUENCE_ENVIRONMENT_VARIABLE) {
        let sequence_path = PathBuf::from(sequence_path);
        let input = kitti_input(&sequence_path).unwrap_or_else(|error| {
            panic!(
                "failed to load KITTI benchmark sequence from {}: {error}",
                sequence_path.display()
            )
        });
        let benchmark_name = format!("kitti_preloaded_{}_frames", input.frames.len());

        group.bench_function(benchmark_name, |bencher| {
            let mut runner = BenchmarkRunner::new(&model_path, &input);

            bencher.iter(|| black_box(runner.step()));
        });
    }

    group.finish();
}

struct BenchmarkRunner<'a> {
    pipeline: Pipeline,
    frames: &'a [StereoFrame],
    next_frame_index: usize,
}

impl<'a> BenchmarkRunner<'a> {
    fn new(model_path: &Path, input: &'a BenchmarkInput) -> Self {
        let pipeline = Pipeline::new(Parameters {
            xfeat_model_path: model_path.to_path_buf(),
            left_calibration: input.left_calibration.clone(),
            right_calibration: input.right_calibration.clone(),
        })
        .unwrap_or_else(|error| panic!("failed to initialize visual odometry pipeline: {error}"));

        Self {
            pipeline,
            frames: &input.frames,
            next_frame_index: 0,
        }
    }

    fn step(&mut self) {
        let frame = &self.frames[self.next_frame_index % self.frames.len()];
        self.next_frame_index += 1;

        self.pipeline
            .step(
                black_box(frame.left_image.view()),
                black_box(frame.right_image.view()),
            )
            .unwrap_or_else(|error| panic!("visual odometry step failed: {error}"));
    }
}

fn model_path() -> PathBuf {
    env::var_os(XFEAT_ENVIRONMENT_VARIABLE)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("xfeat.onnx"))
}

fn synthetic_input(height: usize, width: usize) -> BenchmarkInput {
    BenchmarkInput {
        left_calibration: synthetic_left_calibration(),
        right_calibration: synthetic_right_calibration(),
        frames: vec![StereoFrame {
            left_image: Array2::zeros((height, width)),
            right_image: Array2::zeros((height, width)),
        }],
    }
}

fn synthetic_left_calibration() -> Array2<f32> {
    Array2::from_shape_vec(
        (3, 4),
        vec![
            400.0, 0.0, 320.0, 0.0, 0.0, 400.0, 240.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        ],
    )
    .expect("synthetic calibration has invalid shape")
}

fn synthetic_right_calibration() -> Array2<f32> {
    Array2::from_shape_vec(
        (3, 4),
        vec![
            400.0, 0.0, 320.0, -80.0, 0.0, 400.0, 240.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        ],
    )
    .expect("synthetic calibration has invalid shape")
}

fn kitti_input(sequence_path: &Path) -> Result<BenchmarkInput, Box<dyn Error>> {
    let (left_calibration, right_calibration) =
        load_kitti_calibration(&sequence_path.join("calib.txt"))?;
    let frame_limit = env::var(KITTI_FRAME_LIMIT_ENVIRONMENT_VARIABLE)
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(DEFAULT_KITTI_FRAME_LIMIT);

    let mut frames = Vec::new();
    for frame_index in 0..frame_limit {
        let left_image_path = sequence_path
            .join("image_0")
            .join(format!("{frame_index:06}.png"));
        let right_image_path = sequence_path
            .join("image_1")
            .join(format!("{frame_index:06}.png"));

        if !left_image_path.exists() || !right_image_path.exists() {
            break;
        }

        frames.push(StereoFrame {
            left_image: load_grayscale_image(&left_image_path)?,
            right_image: load_grayscale_image(&right_image_path)?,
        });
    }

    if frames.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "no image pairs found in image_0/image_1",
        )
        .into());
    }

    Ok(BenchmarkInput {
        left_calibration,
        right_calibration,
        frames,
    })
}

fn load_kitti_calibration(path: &Path) -> Result<(Array2<f32>, Array2<f32>), Box<dyn Error>> {
    let content = std::fs::read_to_string(path)?;
    let projections = content
        .lines()
        .map(|line| {
            line.split_whitespace()
                .skip(1)
                .map(str::parse::<f32>)
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;

    if projections.len() < 2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "expected at least two projection matrices, got {}",
                projections.len()
            ),
        )
        .into());
    }

    Ok((
        Array2::from_shape_vec((3, 4), projections[0].clone())?,
        Array2::from_shape_vec((3, 4), projections[1].clone())?,
    ))
}

fn load_grayscale_image(path: &Path) -> Result<Array2<u8>, Box<dyn Error>> {
    let image = ImageReader::open(path)?.decode()?.to_luma8();
    let (width, height) = image.dimensions();

    Ok(Array2::from_shape_vec(
        (height as usize, width as usize),
        image.into_raw(),
    )?)
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(10))
        .sample_size(30);
    targets = benchmark_fps
}
criterion_main!(benches);
