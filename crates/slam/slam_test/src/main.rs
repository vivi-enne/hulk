use std::path::PathBuf;

use clap::Parser;
use color_eyre::eyre::{Context, Result};
use indicatif::{ProgressBarIter, ProgressIterator, ProgressStyle};
use ndarray::s;
use visual_odometry_rust::{VisualOdometryParameters, VisualOdometryPipeline};

use crate::{dataset::KittiOdometrySequence, gui::VisualOdometryGui};

mod dataset;
mod gui;

#[derive(Parser)]
#[clap(version, name = "kitti-example-slam")]
pub struct Arguments {
    /// Path to a KITTI odometry sequence directory.
    sequence: PathBuf,
    /// Path to the Xfeat model file.
    xfeat: PathBuf,
    /// Run in non-interactive mode
    #[arg(long)]
    non_interactive: bool,
}

pub fn main() -> Result<()> {
    color_eyre::install()?;
    let args = Arguments::parse();

    let sequence = KittiOdometrySequence::from_path(args.sequence)?;
    let mut pipeline = VisualOdometryPipeline::new(VisualOdometryParameters {
        xfeat_model_path: args.xfeat,
        left_calibration: sequence.calibration.p0.clone(),
        right_calibration: sequence.calibration.p1.clone(),
    })
    .wrap_err("failed to start visual odometry pipeline")?;

    if args.non_interactive {
        let progress_style =
            ProgressStyle::with_template("{per_sec:<2} {wide_bar:.cyan/blue}").unwrap();

        for i in (0..sequence.len()).progress_with_style(progress_style) {
            let entry = sequence.get(i).unwrap()?;
            pipeline.step(
                entry.left_image.view().slice(s![.., .., 0]),
                entry.right_image.view().slice(s![.., .., 0]),
            )?;
        }
    } else {
        VisualOdometryGui::start(sequence, pipeline).expect("gui failed");
    }

    Ok(())
}
