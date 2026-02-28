use apex_solver::{core::problem::Factor, manifold::se3::SE3};
use nalgebra::{DVector, Point2, Point3, UnitQuaternion, Vector3};

#[derive(Clone, Copy)]
pub struct CameraIntrinsics {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
}

pub struct ReprojectionFactor {
    pub measurement: Point2<f64>,
    pub intrinsics: CameraIntrinsics,
}

impl Factor for ReprojectionFactor {
    // apex_solver factors typically take slices of variable vectors
    // params[0] = pose (7D: tx, ty, tz, qw, qx, qy, qz)
    // params[1] = landmark (3D: x, y, z)
    fn residual(&self, params: &[&DVector<f64>]) -> DVector<f64> {
        let pose_vec = params[0];
        let landmark_vector = params[1];

        // Reconstruct World-to-Camera pose
        let translation = Vector3::new(pose_vec[0], pose_vec[1], pose_vec[2]);
        let quaternion = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
            pose_vec[3],
            pose_vec[4],
            pose_vec[5],
            pose_vec[6],
        ));
        let t_wc = SE3::from_parts(translation, quaternion);

        // Reconstruct 3D Landmark in world frame
        let p_w = Point3::new(landmark_vector[0], landmark_vector[1], landmark_vector[2]);

        // Transform landmark into camera frame: p_c = T_cw * p_w
        // Note: T_cw is the inverse of the camera-to-world pose T_wc
        let p_c = t_wc.inverse() * p_w;

        // Prevent division by zero if point is perfectly at camera center
        let z_c = if p_c.z.abs() < 1e-6 { 1e-6 } else { p_c.z };

        // Project onto image plane
        let u_proj = self.intrinsics.fx * (p_c.x / z_c) + self.intrinsics.cx;
        let v_proj = self.intrinsics.fy * (p_c.y / z_c) + self.intrinsics.cy;

        // Calculate residual: observed - projected
        nalgebra::dvector![self.measurement.x - u_proj, self.measurement.y - v_proj]
    }
}
