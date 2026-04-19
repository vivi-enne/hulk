use std::time::Duration;

use factrs::core::Graph;

use crate::staging::StagingArea;

pub struct BackendConfiguration {
    /// The spacing between control knots on the Gaussian Process
    /// Each control knot represents 9 DoFs for the optimizer.
    pub knot_spacing: Duration,
    /// The maximum optimization window size.
    /// Factors before the optimization window are marginalized.
    pub max_optimization_window: Duration,
}

pub struct LocalizationBackend {
    config: BackendConfiguration,
    graph: Graph,
}

impl LocalizationBackend {
    pub fn new(config: BackendConfiguration) -> Self {
        Self {
            config,
            graph: Graph::new(),
        }
    }

    pub fn consume(&mut self, _staging: StagingArea) {}
}
