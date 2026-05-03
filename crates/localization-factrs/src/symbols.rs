use crate::camera_intrinsics::CameraIntrinsics as CI;
use factrs::{assign_symbols, variables::SE23};

assign_symbols!(State: SE23);
assign_symbols!(CameraIntrinsics: CI);
