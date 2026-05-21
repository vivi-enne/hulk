pub mod simulation_data;

use std::path::PathBuf;

use clap::Parser;
use color_eyre::{Result, eyre::Context};

use crate::simulation_data::SimulationData;

#[derive(Debug, Parser)]
struct Arguments {
    simulation: PathBuf,
}

pub fn main() -> Result<()> {
    let arguments = Arguments::parse();
    let data = std::fs::read_to_string(arguments.simulation)?;
    let data: SimulationData = serde_json::from_str(&data).wrap_err("failed to deserialize")?;

    dbg!(data);

    Ok(())
}
