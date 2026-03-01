use apex_solver::SE3;
use color_eyre::Result;
// mod frontend;

use nalgebra::{Isometry3, Point3, Vector3};
use slam::backend::map::LandmarkMap;
use slam::backend::slam_backend::Backend;

fn main() -> Result<()> {
    let mut backend = Backend::new();
    let mut map = LandmarkMap::new();
    let mut last_pose = Isometry3::identity();

    // simulate random landmarks
    for i in 0..20 {
        map.add_landmark(Point3::new((i * 2) as f64, (i * 3) as f64, 0.0));
    }

    // add initial pose
    let id0 = backend.add_pose(SE3::from_isometry(last_pose.clone()));

    // loop simulate robot motion
    for step in 1..50 {
        // fake odometry motion
        let dx = 0.1;
        let dy = 0.0;
        let dz = 0.0;

        let rot = nalgebra::UnitQuaternion::identity();
        let motion = Isometry3::from_parts(Vector3::new(dx, dy, dz).into(), rot);

        // update pose
        let next_pose = last_pose * motion;

        // create relative transform
        let rel = SE3::from_isometry(motion);

        // add pose and between factor
        let id = backend.add_pose(SE3::from_isometry(next_pose.clone()));
        backend.add_between(id0 + (step - 1) as u64, id0 + step as u64, rel);

        last_pose = next_pose;
    }

    // optimize pose graph
    backend.optimize()?;
    println!("Optimization complete!");
    Ok(())
}
