use std::time::{Duration, Instant, SystemTime};

use booster::ImuState;
use factrs::{core::SO3, traits::Variable, variables::SE23};
use linear_algebra::IntoFramed;
use localization::{
    backend::BackendConfiguration, sparse_gaussian_process::SE23SparseGaussianProcessSegment,
};
use nalgebra::{Matrix3, Vector3, vector};

#[test]
fn imu_on_spline() {
    let (mut frontend, mut backend) = localization::initialize(BackendConfiguration {
        knot_spacing: Duration::from_millis(200),
        max_optimization_window: Duration::from_secs(1),
        gyroscope_noise: Matrix3::identity() * 0.01,
        accelerometer_noise: Matrix3::identity() * 0.1,
        gravity: Vector3::new(0., 0., 9.81),
    });

    let start = SE23::from_rot_vel_trans(SO3::identity(), Vector3::zeros(), Vector3::zeros());
    let end = SE23::from_rot_vel_trans(
        SO3::identity(),
        vector![1.0, 0.0, 0.0],
        vector![2.0, 0.0, 0.0],
    );

    let now = SystemTime::now();
    let ground_truth_spline = SE23SparseGaussianProcessSegment::new(
        now,
        start,
        now + Duration::from_secs(1),
        end,
        Matrix3::identity() * 0.01,
        Matrix3::identity() * 0.1,
    );
    for i in 1..500 {
        let time = now + Duration::from_millis(2) * i;
        let state = ground_truth_spline.infer_derivative(time);

        let imu = ImuState {
            angular_velocity: state
                .fixed_view::<3, 1>(0, 0)
                .into_owned()
                .cast::<f32>()
                .framed(),
            linear_acceleration: state
                .fixed_view::<3, 1>(3, 0)
                .into_owned()
                .cast::<f32>()
                .framed()
                + linear_algebra::vector!(0.0, 0.0, 9.81),
            roll_pitch_yaw: Vector3::zeros().framed(),
        };
        frontend
            .ingest_imu(time, imu)
            .expect("failed to ingest imu");
    }
    let now = Instant::now();
    backend.solve_once().expect("failed to solve");
    let solve_duration = now.elapsed();
    println!("Optimization took {}ms", solve_duration.as_millis());
    let result = frontend.last_optimization_result().expect("no result");
    dbg!(result);
}
