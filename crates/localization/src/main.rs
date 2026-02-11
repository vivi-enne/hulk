use core::num;
use std::{collections::HashMap, fs::File, io::BufWriter};

use color_eyre::Result;
use nalgebra::{Isometry3, Point3, UnitQuaternion, Vector3};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::Distribution;
use serde::Serialize;

#[derive(Serialize)]
struct VisualizeData {
    true_landmarks: Vec<(f32, f32, f32)>,
    history: Vec<VisualizationStep>,
}

#[derive(Serialize)]
struct VisualizationStep {
    robot_particles: Vec<(f32, f32, f32)>,
    estimated_robot_pose: (f32, f32, f32),
    true_robot_pose: (f32, f32, f32),
    particle_landmarks: Vec<Vec<(f32, f32, f32)>>,
    n_eff: f32,
}

struct Observation {
    feature_id: u32,
    position: Vector3<f32>,
}

#[derive(Clone)]
struct PoseParticle {
    id: u32,
    pose: Isometry3<f32>,
    map: HashMap<u32, LandmarkParticleSet>,
    weight: f32,
}

impl PoseParticle {
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
    particles: Vec<PoseParticle>,
    num_particles: u32,
    translation_motion_noise_std: f32,
    rotation_motion_noise_std: f32,
    measurement_noise_std: f32,
    rng: ChaCha8Rng,
}

impl FastSLAM3D {
    pub fn new(num_particles: u32) -> Result<Self> {
        Ok(Self {
            particles: (0..num_particles).map(|i| PoseParticle::new(i)).collect(),
            num_particles,
            translation_motion_noise_std: 0.05, // meters
            rotation_motion_noise_std: 0.03,    // radians
            // Higher measurement noise helps avoid "killing" good particles
            measurement_noise_std: 0.8, // meters
            rng: ChaCha8Rng::seed_from_u64(48),
        })
    }
}

#[derive(Clone)]
struct LandmarkParticle {
    position: Vector3<f32>,
}

#[derive(Clone)]
struct LandmarkParticleSet {
    particles: Vec<LandmarkParticle>,
}

impl LandmarkParticleSet {
    pub fn initialize_landmark_particle_filter(
        pose: &Isometry3<f32>,
        observation: &Observation,
        measurement_noise_std: f32,
        number_particles_per_landmark: usize,
        rng: &mut ChaCha8Rng,
    ) -> LandmarkParticleSet {
        let noise = rand_distr::Normal::new(0.0, measurement_noise_std).unwrap();

        let particles = (0..number_particles_per_landmark)
            .map(|_| {
                // Add noise in the Observation Frame (Range/Bearing implicitly) to represent initial uncertainty of landmark position
                let noisy_observation = observation.position
                    + Vector3::new(noise.sample(rng), noise.sample(rng), noise.sample(rng));

                // Transform to World Frame
                let world_pos = pose * Point3::from(noisy_observation);

                LandmarkParticle {
                    position: world_pos.coords,
                }
            })
            .collect();

        LandmarkParticleSet { particles }
    }

    /// Updates the landmark particles for a single PoseParticle and returns the average likelihood.
    /// If the landmark particles agree with the sensor reading, the likelihood is high.
    /// The corresponding PoseParticle weight is updated with this likelihood.
    pub fn update_and_resample_landmark_particles(
        pose: &Isometry3<f32>,
        landmark: &mut LandmarkParticleSet,
        observation: &Observation,
        measurement_noise_std: f32,
        rng: &mut ChaCha8Rng,
    ) -> f32 {
        let variance = measurement_noise_std * measurement_noise_std;
        // normalization constant for 3D Gaussian
        let normalization_constant = (2.0 * std::f32::consts::PI * variance).powf(-1.5);
        let number_landmark_particles = landmark.particles.len();

        // measurement update
        // checks how well current observation matches each landmark particle
        let mut weights: Vec<f32> = landmark
            .particles
            .iter()
            .map(|landmark_particle| {
                // Predict observation: World -> Robot
                let predicted_local = pose.inverse() * Point3::from(landmark_particle.position);
                let error = observation.position - predicted_local.coords;

                // reweight based on Gaussian Likelihood
                normalization_constant * (-0.5 * error.dot(&error) / variance).exp()
            })
            .collect();

        // high weight indicates good match between observation and landmark particle
        let sum_weights: f32 = weights.iter().sum();

        // avoid division by zero if weights vanish
        if sum_weights < 1e-20 {
            return 1e-20;
        }

        // normalize weights
        weights = weights.iter().map(|weight| weight / sum_weights).collect();

        // resample landmark particles, kills off particles with low weights and duplicates particles with high weights
        let mut new_particles = Vec::with_capacity(number_landmark_particles);

        let step = 1.0 / number_landmark_particles as f32;
        let mut random_offset = rng.random::<f32>() * step;
        let mut cumulative_weight = weights[0];
        let mut i = 0;

        let jitter_std = measurement_noise_std * 0.05;
        let jitter_noise_distribution = rand_distr::Normal::new(0.0, jitter_std).unwrap();

        for _ in 0..number_landmark_particles {
            while random_offset > cumulative_weight {
                i += 1;
                if i >= weights.len() {
                    i = weights.len() - 1;
                }
                cumulative_weight += weights[i];
            }

            // copy particle and add small jitter (to prevent particle collapse)
            let mut particle = landmark.particles[i].clone();
            particle.position += Vector3::new(
                jitter_noise_distribution.sample(rng),
                jitter_noise_distribution.sample(rng),
                jitter_noise_distribution.sample(rng),
            );
            new_particles.push(particle);
            random_offset += step;
        }

        landmark.particles = new_particles;

        // Return average likelihood (Total weight / Num particles)
        // This is the weight update for the ROBOT particle
        sum_weights / number_landmark_particles as f32
    }
}

struct ParticleCloud {
    particles: Vec<PoseParticle>,
}

impl ParticleCloud {
    /// Predictes movement by adding noise to create a cloud of possible new positions based on odometry
    pub fn predict_particles(
        particles: &mut Vec<PoseParticle>,
        delta_odometry: Isometry3<f32>,
        translation_noise_std: f32,
        rotation_noise_std: f32,
        rng: &mut ChaCha8Rng,
    ) {
        let translation_noise_distribution =
            rand_distr::Normal::new(0.0, translation_noise_std).unwrap();
        let rotation_noise_distribution = rand_distr::Normal::new(0.0, rotation_noise_std).unwrap();

        particles.iter_mut().for_each(|particle| {
            let translation_noise = Vector3::new(
                translation_noise_distribution.sample(rng),
                translation_noise_distribution.sample(rng),
                translation_noise_distribution.sample(rng),
            );

            let rotation_noise = Vector3::new(
                rotation_noise_distribution.sample(rng),
                rotation_noise_distribution.sample(rng),
                rotation_noise_distribution.sample(rng),
            );

            let angle = rotation_noise.norm();
            let rot_noise_q = if angle > 1e-6 {
                nalgebra::UnitQuaternion::from_axis_angle(
                    &nalgebra::Unit::new_normalize(rotation_noise),
                    angle,
                )
            } else {
                nalgebra::UnitQuaternion::identity()
            };

            let noisy_delta = Isometry3::from_parts(
                (delta_odometry.translation.vector + translation_noise).into(),
                delta_odometry.rotation * rot_noise_q,
            );

            particle.pose = particle.pose * noisy_delta;
        });
    }

    // Update particle weights with a single observation, use importance sampling to determine new particle weights
    pub fn update_particles_with_observation(
        particles: &mut Vec<PoseParticle>,
        observation: Observation,
        measurement_noise_std: f32,
        number_landmark_particles: usize,
        rng: &mut ChaCha8Rng,
    ) {
        particles.iter_mut().for_each(|particle| {
            if let Some(landmark) = particle.map.get_mut(&observation.feature_id) {
                // Update existing landmark particles
                let likelihood = LandmarkParticleSet::update_and_resample_landmark_particles(
                    &particle.pose,
                    landmark,
                    &observation,
                    measurement_noise_std,
                    rng,
                );
                particle.weight *= likelihood;
            } else {
                // Initialize new landmark particles
                let landmark = LandmarkParticleSet::initialize_landmark_particle_filter(
                    &particle.pose,
                    &observation,
                    measurement_noise_std,
                    number_landmark_particles,
                    rng,
                );
                particle.map.insert(observation.feature_id, landmark);
                // New landmark weight is neutral
            }
        });
    }

    // Resample particles if weight drops under threshold to avoid particle depletion (single particle dominates weight and filter fails)
    pub fn resample_particles(particles: &mut Vec<PoseParticle>, rng: &mut ChaCha8Rng) {
        let number_particles = particles.len();
        if number_particles == 0 {
            return;
        }

        Self::normalize_weights(particles);

        let sum_squared_weights: f32 = particles.iter().map(|p| p.weight.powi(2)).sum();
        let effective_number_samples = 1.0 / sum_squared_weights;

        // Adaptive Resampling to avoid particle depletion: Threshold 2N/3
        if effective_number_samples > (number_particles as f32 * 2.0 / 3.0) {
            return;
        }

        let step = 1.0 / number_particles as f32;
        let mut random_offset = rng.random::<f32>() * step;
        let mut cumulative_weight = particles[0].weight;
        let mut i = 0;

        let mut new_particles = Vec::with_capacity(number_particles);

        // Low Variance Resampling
        for _ in 0..number_particles {
            // advances to next particle index i until cumulative weight exceeds random offset
            while random_offset > cumulative_weight {
                i += 1;
                if i >= number_particles {
                    i = number_particles - 1;
                    break;
                }
                cumulative_weight += particles[i].weight;
            }
            let mut particle = particles[i].clone();
            particle.weight = 1.0 / number_particles as f32;
            // adds this particle to the new set, particles with high weights are selected a proportional number of times
            new_particles.push(particle);
            random_offset += step;
        }
        *particles = new_particles;
    }

    pub fn normalize_weights(particles: &mut Vec<PoseParticle>) {
        let sum_weights: f32 = particles.iter().map(|particle| particle.weight).sum();
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

    pub fn estimate_pose(particles: &[PoseParticle]) -> Isometry3<f32> {
        let mut position = Vector3::zeros();
        let mut sin_yaw = 0.0;
        let mut cos_yaw = 0.0;
        let mut total_weight = 0.0;

        for particle in particles {
            position += particle.pose.translation.vector * particle.weight;

            let yaw = particle.pose.rotation.euler_angles().2;
            sin_yaw += yaw.sin() * particle.weight;
            cos_yaw += yaw.cos() * particle.weight;
            total_weight += particle.weight;
        }

        if total_weight > 0.0 {
            position /= total_weight;
            sin_yaw /= total_weight;
            cos_yaw /= total_weight;
        }

        let yaw = sin_yaw.atan2(cos_yaw);
        let rotation = nalgebra::UnitQuaternion::from_euler_angles(0.0, 0.0, yaw);
        Isometry3::from_parts(position.into(), rotation)
    }
}

fn main() {
    let mut slam = FastSLAM3D::new(500).expect("SLAM init failed");

    // Constants
    const DT: f32 = 0.1;
    const SIMULATION_TIME: f32 = 300.0;
    const NUMBER_LANDMARK_PARTICLES: usize = 10;

    // // True Landmarks
    // let landmark_positions = vec![
    //     Vector3::new(10.0, -2.0, 0.0),
    //     Vector3::new(15.0, 10.0, 0.0),
    //     Vector3::new(15.0, 15.0, 0.0),
    //     Vector3::new(10.0, 20.0, 0.0),
    //     Vector3::new(3.0, 15.0, 0.0),
    //     Vector3::new(-5.0, 20.0, 0.0),
    //     Vector3::new(-5.0, 5.0, 0.0),
    //     Vector3::new(-10.0, 15.0, 0.0),
    // ];

    // 3D Landmark Positions (Helix pattern)
    let landmark_positions = vec![
        Vector3::new(10.0, -2.0, 0.0),
        Vector3::new(15.0, 10.0, 2.0),
        Vector3::new(15.0, 15.0, 4.0),
        Vector3::new(10.0, 20.0, 6.0),
        Vector3::new(3.0, 15.0, 8.0),
        Vector3::new(-5.0, 20.0, 6.0),
        Vector3::new(-5.0, 5.0, 4.0),
        Vector3::new(-10.0, 15.0, 2.0),
        Vector3::new(0.0, 0.0, 10.0), // High central landmark
    ];

    let mut particles = slam.particles.clone();
    let mut true_robot_pose = Isometry3::identity();
    let mut history: Vec<VisualizationStep> = Vec::new();
    let mut prev_est_pose = Vector3::zeros();
    let mut step_count = 0;

    let bar = indicatif::ProgressBar::new((SIMULATION_TIME / DT) as u64);

    for time in (0..(SIMULATION_TIME / DT) as u32).map(|x| x as f32 * DT) {
        bar.inc(1);
        step_count += 1;

        // 3D Control Input (Upward Spiral)
        let (v_forward, v_up, yaw_rate) = if time <= 1.0 {
            (0.0, 0.0, 0.0)
        } else {
            (1.0, 0.05, 0.1) // Move forward, slowly up, and turn
        };

        // Create 3D motion command
        let delta_trans = Vector3::new(v_forward * DT, 0.0, v_up * DT); // Robot frame: x=forward, z=up
        let delta_rot = UnitQuaternion::from_euler_angles(0.0, 0.0, yaw_rate * DT);
        let motion = Isometry3::from_parts(delta_trans.into(), delta_rot);

        // Update Truth
        true_robot_pose = true_robot_pose * motion;

        // Predict (Motion Model)
        ParticleCloud::predict_particles(
            &mut particles,
            motion,
            slam.translation_motion_noise_std,
            slam.rotation_motion_noise_std,
            &mut slam.rng,
        );

        //  Observations (Simulated)
        let observations: Vec<Observation> = landmark_positions
            .iter()
            .enumerate()
            .map(|(id, lm)| {
                let obs_pos = true_robot_pose.inverse() * Point3::from(*lm);
                Observation {
                    feature_id: id as u32,
                    position: obs_pos.coords,
                }
            })
            .collect();

        // Update
        for obs in observations {
            ParticleCloud::update_particles_with_observation(
                &mut particles,
                obs,
                slam.measurement_noise_std,
                NUMBER_LANDMARK_PARTICLES,
                &mut slam.rng,
            );
        }

        // Need temporary normalization to calculate the stat correctly
        // (This doesn't change the relative weights, just scales them to sum to 1)
        let sum_weights: f32 = particles.iter().map(|p| p.weight).sum();
        let current_n_eff = if sum_weights > 0.0 {
            let sum_sq: f32 = particles
                .iter()
                .map(|particle| (particle.weight / sum_weights).powi(2))
                .sum();
            1.0 / sum_sq
        } else {
            0.0
        };

        // Resample
        ParticleCloud::resample_particles(&mut particles, &mut slam.rng);

        // Estimate & Visualize
        let pose = ParticleCloud::estimate_pose(&particles);
        let current_pos_vec = pose.translation.vector;
        let jump_dist = (current_pos_vec - prev_est_pose).norm();
        if jump_dist > 0.5 && time > 2.0 {
            println!(
                "Jump! Step {} (Time {:.1}s), dist {:.3}m",
                step_count, time, jump_dist
            );
        }
        prev_est_pose = current_pos_vec;

        // Visualization Export
        let robot_particles: Vec<(f32, f32, f32)> = particles
            .iter()
            .map(|particle| {
                (
                    particle.pose.translation.x,
                    particle.pose.translation.y,
                    particle.pose.translation.z,
                )
            })
            .collect();

        // Extract Landmark Means (Average the particles for viz)
        let particle_landmarks: Vec<Vec<(f32, f32, f32)>> = particles
            .iter()
            .map(|p| {
                p.map
                    .values()
                    .map(|landmark_set| {
                        let sum: Vector3<f32> =
                            landmark_set.particles.iter().map(|lp| lp.position).sum();
                        let mean = sum / (landmark_set.particles.len() as f32);
                        (mean.x, mean.y, mean.z)
                    })
                    .collect()
            })
            .collect();

        history.push(VisualizationStep {
            robot_particles,
            estimated_robot_pose: (pose.translation.x, pose.translation.y, pose.translation.z),
            true_robot_pose: (
                true_robot_pose.translation.x,
                true_robot_pose.translation.y,
                true_robot_pose.translation.z,
            ),
            particle_landmarks,
            n_eff: current_n_eff,
        });
    }
    bar.finish();

    let data = VisualizeData {
        true_landmarks: landmark_positions.iter().map(|l| (l.x, l.y, l.z)).collect(),
        history,
    };

    let file = File::create("slam_data.json").expect("Create file failed");
    let writer = BufWriter::new(file);
    serde_json::to_writer(writer, &data).expect("Write JSON failed");
    println!("Written slam_data.json");
}
