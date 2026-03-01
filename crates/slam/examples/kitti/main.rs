use std::path::PathBuf;

use clap::Parser;
use color_eyre::eyre::Result;

use crate::{dataset::KittiOdometrySequence, gui::VisualOdometryGui};

mod dataset;
mod gui;
mod visual_odometry;

#[derive(Parser)]
#[clap(version, name = "kitti-example-slam")]
pub struct Arguments {
    path: PathBuf,
}

pub fn main() -> Result<()> {
    color_eyre::install()?;
    let args = Arguments::parse();

    let sequence = KittiOdometrySequence::from_path(args.path)?;
    VisualOdometryGui::start(sequence).expect("gui failed");

    Ok(())
}
