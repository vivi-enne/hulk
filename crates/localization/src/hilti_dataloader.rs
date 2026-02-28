use serde::Deserialize;
use std::error::Error;

#[derive(Debug, Deserialize)]
struct HiltiGroundTruth {
    #[serde(rename = "#sec")]
    sec: u64,
    #[serde(rename = "nsec")]
    nsec: u64,
    x: f32, y: f32, z: f32,
    qx: f32, qy: f32, qz: f32, qw: f32,
}

// Minimalist loader for the Hilti Benchmarking files
pub fn load_hilti_gt(path: &str) -> Vec<HiltiGroundTruth> {
    let mut rdr = csv::Reader::from_path(path).expect("Failed to open Hilti CSV");
    rdr.deserialize().filter_map(|result| result.ok()).collect()
}