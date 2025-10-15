use std::{
    iter::Sum,
    ops::{Add, Mul},
};

use color_eyre::{
    eyre::{eyre, WrapErr},
    Report, Result,
};
use serde::{Deserialize, Serialize};
use serde_json::{from_value, Value};

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct HeadJoints {
    pub yaw: f32,
    pub pitch: f32,
}

impl From<Joints> for HeadJoints {
    fn from(joints: Joints) -> Self {
        Self {
            yaw: joints.head.yaw,
            pitch: joints.head.pitch,
        }
    }
}

impl Mul<f32> for HeadJoints {
    type Output = HeadJoints;

    fn mul(self, scale_factor: f32) -> Self::Output {
        Self::Output {
            yaw: self.yaw * scale_factor,
            pitch: self.pitch * scale_factor,
        }
    }
}

impl Add for HeadJoints {
    type Output = HeadJoints;

    fn add(self, right: Self) -> Self::Output {
        Self::Output {
            yaw: self.yaw + right.yaw,
            pitch: self.pitch + right.pitch,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct ArmJoints {
    pub shoulder_pitch: f32,
    pub shoulder_roll: f32,
    pub shoulder_yaw: f32,
    pub elbow: f32,
}

impl Mul<f32> for ArmJoints {
    type Output = ArmJoints;

    fn mul(self, scale_factor: f32) -> Self::Output {
        Self::Output {
            shoulder_pitch: self.shoulder_pitch * scale_factor,
            shoulder_roll: self.shoulder_roll * scale_factor,
            shoulder_yaw: self.shoulder_yaw * scale_factor,
            elbow: self.elbow * scale_factor,
        }
    }
}

impl Add for ArmJoints {
    type Output = ArmJoints;

    fn add(self, right: Self) -> Self::Output {
        Self::Output {
            shoulder_pitch: self.shoulder_pitch + right.shoulder_pitch,
            shoulder_roll: self.shoulder_roll + right.shoulder_roll,
            shoulder_yaw: self.shoulder_yaw + right.shoulder_yaw,
            elbow: self.elbow + right.elbow,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct LegJoints {
    pub hip_pitch: f32,
    pub hip_roll: f32,
    pub hip_yaw: f32,
    pub knee: f32,
    pub ankle_up: f32,
    pub ankle_down: f32,
}

impl Mul<f32> for LegJoints {
    type Output = LegJoints;

    fn mul(self, scale_factor: f32) -> Self::Output {
        Self::Output {
            hip_pitch: self.hip_pitch * scale_factor,
            hip_roll: self.hip_roll * scale_factor,
            hip_yaw: self.hip_yaw * scale_factor,
            knee: self.knee * scale_factor,
            ankle_up: self.ankle_up * scale_factor,
            ankle_down: self.ankle_down * scale_factor,
        }
    }
}

impl Add for LegJoints {
    type Output = LegJoints;

    fn add(self, right: Self) -> Self::Output {
        Self::Output {
            hip_yaw: self.hip_yaw + right.hip_yaw,
            hip_roll: self.hip_roll + right.hip_roll,
            hip_pitch: self.hip_pitch + right.hip_pitch,
            knee: self.knee + right.knee,
            ankle_up: self.ankle_up + right.ankle_up,
            ankle_down: self.ankle_down + right.ankle_down,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct BodyJoints {
    pub left_arm: ArmJoints,
    pub right_arm: ArmJoints,
    pub left_leg: LegJoints,
    pub right_leg: LegJoints,
}

impl From<Joints> for BodyJoints {
    fn from(joints: Joints) -> Self {
        Self {
            left_arm: joints.left_arm,
            right_arm: joints.right_arm,
            left_leg: joints.left_leg,
            right_leg: joints.right_leg,
        }
    }
}

impl Mul<f32> for BodyJoints {
    type Output = BodyJoints;

    fn mul(self, scale_factor: f32) -> Self::Output {
        Self::Output {
            left_arm: self.left_arm * scale_factor,
            right_arm: self.right_arm * scale_factor,
            left_leg: self.left_leg * scale_factor,
            right_leg: self.right_leg * scale_factor,
        }
    }
}

impl Add for BodyJoints {
    type Output = BodyJoints;

    fn add(self, right: Self) -> Self::Output {
        Self::Output {
            left_arm: self.left_arm + right.left_arm,
            right_arm: self.right_arm + right.right_arm,
            left_leg: self.left_leg + right.left_leg,
            right_leg: self.right_leg + right.right_leg,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct Joints {
    pub head: HeadJoints,
    pub left_arm: ArmJoints,
    pub right_arm: ArmJoints,
    pub left_leg: LegJoints,
    pub right_leg: LegJoints,
}

impl TryFrom<&Value> for Joints {
    type Error = Report;

    fn try_from(replay_frame: &Value) -> Result<Self> {
        let joint_angles = replay_frame
            .get("jointAngles")
            .ok_or_else(|| eyre!("replay_frame.get(\"jointAngles\")"))?;
        let angles: Vec<f32> =
            from_value(joint_angles.clone()).wrap_err("from_value(joint_angles)")?;
        Ok(Self::from_angles(&angles))
    }
}

impl Joints {
    pub fn from_angles(angles: &[f32]) -> Self {
        assert!(angles.len() >= 26);
        Self {
            head: HeadJoints {
                yaw: angles[0],
                pitch: angles[1],
            },
            left_arm: ArmJoints {
                shoulder_pitch: angles[2],
                shoulder_roll: angles[3],
                shoulder_yaw: angles[4],
                elbow: angles[5],
            },
            right_arm: ArmJoints {
                shoulder_pitch: angles[6],
                shoulder_roll: angles[7],
                shoulder_yaw: angles[8],
                elbow: angles[9],
            },
            left_leg: LegJoints {
                hip_pitch: angles[10],
                hip_roll: angles[11],
                hip_yaw: angles[12],
                knee: angles[13],
                ankle_up: angles[14],
                ankle_down: angles[15],
            },
            right_leg: LegJoints {
                hip_pitch: angles[16],
                hip_roll: angles[17],
                hip_yaw: angles[18],
                knee: angles[19],
                ankle_up: angles[20],
                ankle_down: angles[21],
            },
        }
    }
}

impl Mul<f32> for Joints {
    type Output = Joints;

    fn mul(self, scale_factor: f32) -> Self::Output {
        Self::Output {
            head: self.head * scale_factor,
            left_arm: self.left_arm * scale_factor,
            right_arm: self.right_arm * scale_factor,
            left_leg: self.left_leg * scale_factor,
            right_leg: self.right_leg * scale_factor,
        }
    }
}

impl Add for Joints {
    type Output = Joints;

    fn add(self, right: Self) -> Self::Output {
        Self::Output {
            head: self.head + right.head,
            left_arm: self.left_arm + right.left_arm,
            right_arm: self.right_arm + right.right_arm,
            left_leg: self.left_leg + right.left_leg,
            right_leg: self.right_leg + right.right_leg,
        }
    }
}

impl Sum for Joints {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Joints::default(), |acc, x| acc + x)
    }
}
