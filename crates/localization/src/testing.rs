use std::{collections::HashMap, hash::Hash};

use color_eyre::Result;
use linear_algebra::{Isometry3, Matrix3, Point3, Vector3};
use rayon::prelude::*;

//TODO: move
struct Feature {
    id: u32,
    position: Vector3<f32>,
}

//TODO: move
struct Observation {
    feature_id: u32,
    /// Relative position (in robot frame)
    position: Vector3<f32>,
}

struct LandmarkEstimate {
    position: Vector3<f32>,
    covariance: Matrix3<f32>,
    num_observations: u32,
}

struct Particle {
    id: u32,
    pose: Isometry3<f32>,
    map: HashMap<u32, LandmarkEstimate>,
    weight: f32,
}

impl Particle {
    pub fn new(id: u32) -> Self {
        Self {
            id,
            pose: Isometry3::identity(),
            map: HashMap::new(),
            weight: 1.0,
        }
    }
}

struct FastSLAM3D {
    particles: Vec<Particle>,
    num_particles: u32,

    translation_motion_noise_std: f32,
    rotation_motion_noise_std: f32,
    measurement_noise_std: f32,
}

impl FastSLAM3D {
    pub fn new(num_particles: u32) -> Result<Self> {
        Ok(Self {
            particles: (0..num_particles).map(|i| Particle::new(i)).collect(),
            num_particles,
            translation_motion_noise_std: 0.1, // meters
            rotation_motion_noise_std: 0.05,   // radians
            measurement_noise_std: 0.2,        // meters
        })
    }

    /// Observe a new feature and add it to the map
    pub fn observe_new(&self, feature: Feature) {
        self.map.push(feature);
    }

    //TODO: what is u? is it the noise?
    pub fn predict_particles(&mut self, delta_odom: Isometry3<f32>) {
        use rand_distr::Distribution;

        let trans_noise = rand_distr::Normal::new(0.0, self.translation_motion_noise_std).unwrap();
        let rot_noise = rand_distr::Normal::new(0.0, self.rotation_motion_noise_std).unwrap();

        self.particles.iter_mut().for_each(|p| {
            let mut rng = rand::thread_rng();

            // Translation noise
            let noise_t = Vector3::new(
                trans_noise.sample(&mut rng),
                trans_noise.sample(&mut rng),
                trans_noise.sample(&mut rng),
            );

            // Rotation noise (small-angle approximation)
            let axis = Vector3::new(
                rot_noise.sample(&mut rng),
                rot_noise.sample(&mut rng),
                rot_noise.sample(&mut rng),
            );

            let angle = axis.norm();
            let rot_noise_q = if angle > 1e-6 {
                nalgebra::UnitQuaternion::from_axis_angle(
                    &nalgebra::Unit::new_normalize(axis),
                    angle,
                )
            } else {
                nalgebra::UnitQuaternion::identity()
            };

            let noisy_delta = Isometry3::from_parts(
                (delta_odom.translation.vector + noise_t).into(),
                delta_odom.rotation * rot_noise_q,
            );

            p.pose = p.pose * noisy_delta;
        });
    }

    fn initialize_landmark(
        particle_pose: &Isometry3<f32>,
        obs: &Observation,
        meas_noise: f32,
    ) -> LandmarkEstimate {
        let world_pos = particle_pose * Point3::from(obs.position);

        LandmarkEstimate {
            position: world_pos.coords,
            covariance: Matrix3::identity() * meas_noise.powi(2),
            num_observations: 1,
        }
    }

    fn update_landmark_ekf(
        particle_pose: &Isometry3<f32>,
        landmark: &mut LandmarkEstimate,
        obs: &Observation,
        meas_noise: f32,
    ) -> f32 {
        // Predicted observation
        let landmark_world = Point3::from(landmark.position);
        let predicted = particle_pose.inverse() * landmark_world;

        let innovation = obs.position - predicted.coords;

        // Jacobian H ≈ Identity (small-angle approx, point landmark)
        let h = Matrix3::identity();
        let r = Matrix3::identity() * meas_noise.powi(2);

        let s = h * landmark.covariance * h.transpose() + r;
        let k = landmark.covariance * h.transpose() * s.try_inverse().unwrap();

        // Update
        landmark.position += k * innovation;
        landmark.covariance = (Matrix3::identity() - k * h) * landmark.covariance;
        landmark.num_observations += 1;

        // Likelihood for particle weight
        let mahalanobis = innovation.transpose() * s.try_inverse().unwrap() * innovation;
        (-0.5 * mahalanobis[(0, 0)]).exp()
    }

    pub fn update_particles_with_observation(&mut self, observation: Observation) {
        self.particles.iter_mut().for_each(|p| {
            let likelihood = if let Some(landmark) = p.map.get_mut(&observation.feature_id) {
                update_landmark_ekf(&p.pose, landmark, &observation, self.measurement_noise_std)
            } else {
                let landmark =
                    initialize_landmark(&p.pose, &observation, self.measurement_noise_std);
                p.map.insert(observation.feature_id, landmark);
                1.0
            };

            p.weight *= likelihood;
        });
    }

    pub fn resample_particles(&mut self) {
        self.normalize_weights();

        let mut rng = rand::thread_rng();
        let n = self.particles.len();
        let step = 1.0 / n as f32;
        let mut r = rng.gen::<f32>() * step;

        let mut c = self.particles[0].weight;
        let mut i = 0;

        let mut new_particles = Vec::with_capacity(n);

        for _ in 0..n {
            while r > c {
                i += 1;
                c += self.particles[i].weight;
            }

            let mut p = self.particles[i].clone();
            p.weight = 1.0 / n as f32;
            new_particles.push(p);

            r += step;
        }

        self.particles = new_particles;
    }

    pub fn normalize_weights(mut particles: Vec<Particle>) {
        let sum_weights = particles.iter().map(|particle| particle.weight).sum();
        if sum_weights > 0.0 {
            particles
                .iter_mut()
                .for_each(|particle| particle.weight /= sum_weights);
        } else {
            // recovery if all particles died
            let uniform_weight = 1.0 / particles.len() as f32;
            particles
                .iter_mut()
                .for_each(|particle| particle.weight = uniform_weight);
        }
    }

}
