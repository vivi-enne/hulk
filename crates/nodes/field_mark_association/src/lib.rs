use std::{
    future::{Future, ready},
    pin::Pin,
    sync::Arc,
    time::Duration,
};

use color_eyre::{Result, eyre::Context as _};
use coordinate_systems::{Camera, Field, Pixel, Robot};
use global_association::{
    FeatureAssociation, GlobalAssociator, GlobalLocalizationInput, GlobalLocalizationResult,
    PoseHintAssociationResult,
};
use linear_algebra::{Isometry3, Point2, Point3, point};
use projection::camera_matrix::CameraMatrix;
use ros_z::{Message, context::Context, parameter::NodeParametersExt, qos::QosDurability};
use ros_z_streams::CreateFutureMapBuilder;
use serde::{Deserialize, Serialize};
use types::{
    field_dimensions::FieldDimensions,
    object_detection::{Object, RobocupObjectLabel},
    time_wrapper::TimeWrapper,
};

mod global_association;

pub use global_association::{
    GlobalAssociationConfig as GlobalLocalizerParameters, GlobalLocalizationDebugAssociation,
    GlobalLocalizationDebugDetection, GlobalLocalizationDebugProjection,
    GlobalLocalizationDetailedDebug, GlobalLocalizationDetailedStatus, GlobalLocalizationScore,
    PoseHintAssociationConfig as PoseHintAssociationParameters, VisualFeatureClass,
};

const MAX_CAMERA_MATRIX_TIME_DISTANCE: Duration = Duration::from_millis(100);
const DETECTED_OBJECTS_SAFETY_LAG: Duration = Duration::from_millis(50);

#[derive(Clone, Debug, Deserialize, Serialize, Message)]
#[serde(deny_unknown_fields)]
pub struct FieldMarkAssociationParameters {
    pub global_localizer: GlobalLocalizerParameters,
    pub pose_hint: PoseHintAssociationParameters,
}

impl Default for FieldMarkAssociationParameters {
    fn default() -> Self {
        Self {
            global_localizer: GlobalLocalizerParameters::default(),
            pose_hint: PoseHintAssociationParameters::default(),
        }
    }
}

impl FieldMarkAssociationParameters {
    fn validate(&self) -> std::result::Result<(), String> {
        self.global_localizer.validate()?;
        self.pose_hint.validate()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Message)]
pub struct FieldMarkAssociation {
    pub detection: Point2<Pixel>,
    pub field_point: Point3<Field>,
    pub kind: FieldMarkAssociationKind,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize, Message)]
pub enum FieldMarkAssociationKind {
    GlobalUnique,
    PoseHint,
}

#[derive(Debug, Clone, Serialize, Deserialize, Message)]
pub struct FieldMarkAssociations {
    pub robot_to_camera: Isometry3<Robot, Camera>,
    pub associations: Vec<FieldMarkAssociation>,
}

/// Result of running visual global localization on one object-detection frame.
pub struct GlobalVisualLocalization {
    /// Debug payload for the best visual global localization result, if any.
    pub debug: Option<GlobalLocalizationDebug>,
    /// Fixed associations selected by either global uniqueness or pose-hint fallback.
    pub associations: Vec<FieldMarkAssociation>,
}

#[derive(Debug, Clone, Default)]
pub struct FieldMarkAssociationState {
    weak_pose_hint_frames: usize,
    pending_global_recovery: Option<GlobalRecoveryCandidate>,
    has_global_lock: bool,
}

#[derive(Debug, Clone)]
struct GlobalRecoveryCandidate {
    robot_to_field: Isometry3<Robot, Field>,
    associations: Vec<FieldMarkAssociation>,
    consecutive_frames: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Message)]
/// Published debug data for a successful global localization hypothesis.
pub struct GlobalLocalizationDebug {
    /// Best robot pose in the field frame for this visual result.
    pub robot_to_field: Isometry3<Robot, Field>,
    /// Whether the best result is ambiguous or unique modulo field symmetry.
    pub status: GlobalLocalizationDebugStatus,
    /// Number of fixed feature associations accepted by the global-localizer gates.
    pub inliers: usize,
    /// Root-mean-square reprojection error in pixels.
    pub reprojection_rmse: f32,
    /// Sum of squared reprojection errors in pixels squared.
    pub total_cost: f32,
    /// Weighted internal candidate score used by `min_score` and `score_ratio`.
    pub candidate_score: f32,
    /// Metric field-space RMS residual used by `rms_threshold`.
    pub metric_rms_residual: f32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Message)]
/// Classification of a successful global localization result.
pub enum GlobalLocalizationDebugStatus {
    /// Uniqueness was not certified because a competitor may remain or the bounded search ended
    /// inconclusively.
    Ambiguous,
    /// Reserved old wire tag for the removed strict-unique status. New code must not emit it.
    #[deprecated(note = "strict unique is no longer emitted; use UniqueModuloSymmetry")]
    Unique,
    /// The assignment is unique after quotienting the unavoidable 180 degree
    /// field symmetry. The chosen branch follows the pose hint when available.
    UniqueModuloSymmetry,
}

/// Starts the field-mark association node and erases the concrete future type for node runners.
pub fn run_boxed(ctx: Arc<Context>) -> Pin<Box<dyn Future<Output = Result<()>> + Send>> {
    Box::pin(run(ctx))
}

pub async fn run(ctx: Arc<Context>) -> Result<()> {
    let node = ctx.create_node("field_mark_association").build().await?;
    let parameters =
        node.bind_parameter_as::<FieldMarkAssociationParameters>("field_mark_association")?;
    parameters.add_validation_hook(FieldMarkAssociationParameters::validate)?;

    let camera_matrix_cache = node
        .create_cache::<TimeWrapper<CameraMatrix>>("camera_matrix", 128)?
        .with_stamp(|message| message.time)
        .build()
        .await?;

    let field_dimensions_cache = node
        .create_cache::<FieldDimensions>("field_dimensions", 1)?
        .with_qos(ros_z::qos::QosProfile {
            durability: QosDurability::TransientLocal,
            ..Default::default()
        })
        .build()
        .await?;

    let localization_cache = node
        .create_cache::<TimeWrapper<Option<Isometry3<Field, Robot>>>>(
            "localization/timestamped",
            128,
        )?
        .with_stamp(|message| message.time)
        .build()
        .await?;

    let mut detected_objects = node
        .create_future_map_builder()
        .create_future_subscriber::<Vec<Object<RobocupObjectLabel>>>(
            "detected_objects",
            DETECTED_OBJECTS_SAFETY_LAG,
        )
        .await?
        .build();

    let associations_publisher = node
        .publisher::<TimeWrapper<FieldMarkAssociations>>("field_mark_association/associations")?
        .build()
        .await?;
    let global_localization_publisher = node
        .publisher::<Option<GlobalLocalizationDebug>>("debug/global_localization")?
        .build()
        .await?;
    let mut association_state = FieldMarkAssociationState::default();

    loop {
        let item = detected_objects.recv().await?;
        for (image_time, (objects,)) in item.persistent {
            let Some(camera_matrix) = camera_matrix_cache.get_nearest(image_time) else {
                continue;
            };
            if !camera_matrix_is_fresh(&camera_matrix, image_time) {
                continue;
            }
            let Some(field_dimensions) = field_dimensions_cache.get_nearest(image_time) else {
                continue;
            };

            let parameters = parameters.snapshot().typed().clone();
            let objects = objects.unwrap_or_default();
            let camera_matrix = camera_matrix.inner.clone();
            let robot_to_camera = robot_to_camera(&camera_matrix);
            let field_dimensions = *field_dimensions.as_ref();
            let pose_hint = localization_cache
                .get_nearest_with_stamp(image_time)
                .and_then(|(stamp, localization)| {
                    (time_distance(stamp, image_time) <= parameters.pose_hint.max_pose_age)
                        .then(|| {
                            localization
                                .inner
                                .as_ref()
                                .map(|pose| pose.clone().inverse())
                        })
                        .flatten()
                });
            let include_debug = global_localization_publisher.has_subscribers();

            let mut state = std::mem::take(&mut association_state);
            let (state, localization) = tokio::task::spawn_blocking(move || {
                let visual_features = find_detected_visual_features(&objects);
                if visual_features.supported_feature_count() == 0 {
                    return (
                        state,
                        GlobalVisualLocalization {
                            debug: None,
                            associations: Vec::new(),
                        },
                    );
                }

                let localization = state.associate_visual_features_with_debug(
                    &visual_features,
                    &camera_matrix,
                    &field_dimensions,
                    pose_hint,
                    &parameters,
                    include_debug,
                );
                (state, localization)
            })
            .await
            .wrap_err("field mark association task failed")?;
            association_state = state;

            let debug = localization.debug.clone();
            global_localization_publisher
                .publish_if_subscribed(|| ready(debug))
                .await?;

            let message = TimeWrapper {
                time: image_time,
                inner: FieldMarkAssociations {
                    robot_to_camera,
                    associations: localization.associations,
                },
            };
            associations_publisher.publish(&message).await?;
        }
    }
}

fn camera_matrix_is_fresh(
    camera_matrix: &TimeWrapper<CameraMatrix>,
    time: ros_z::time::Time,
) -> bool {
    time_distance(camera_matrix.time, time) <= MAX_CAMERA_MATRIX_TIME_DISTANCE
}

fn time_distance(a: ros_z::time::Time, b: ros_z::time::Time) -> Duration {
    Duration::from_nanos(a.as_nanos().abs_diff(b.as_nanos()))
}

/// Runs global localization first and falls back to pose-hint association when needed.
pub fn associate_visual_features(
    visual_features: &DetectedVisualFeatures,
    camera_matrix: &CameraMatrix,
    field_dimensions: &FieldDimensions,
    pose_hint: Option<Isometry3<Robot, Field>>,
    parameters: &FieldMarkAssociationParameters,
) -> GlobalVisualLocalization {
    let localizer = GlobalAssociator::new(parameters.global_localizer);
    let input = GlobalLocalizationInput {
        visual_features,
        field_dimensions,
        ground_to_robot: camera_matrix.ground_to_robot,
        robot_to_camera: robot_to_camera(camera_matrix),
        camera_intrinsic: camera_matrix.intrinsics,
        pose_hint,
    };

    let pose_hint_result = localizer.associate_with_pose_hint(input.clone(), parameters.pose_hint);
    if !pose_hint_result.associations.is_empty() {
        return GlobalVisualLocalization {
            debug: None,
            associations: pose_hint_field_mark_associations(&pose_hint_result),
        };
    }

    let result = localizer.localize(input);
    GlobalVisualLocalization {
        debug: result.as_ref().map(global_localization_debug_from_result),
        associations: result
            .as_ref()
            .and_then(GlobalLocalizationResult::unique_feature_associations)
            .map(|associations| {
                field_mark_associations(associations.iter(), FieldMarkAssociationKind::GlobalUnique)
            })
            .unwrap_or_default(),
    }
}

impl FieldMarkAssociationState {
    /// Associates visual field features while preserving pose-hint health and global recovery state.
    pub fn associate_visual_features_with_debug(
        &mut self,
        visual_features: &DetectedVisualFeatures,
        camera_matrix: &CameraMatrix,
        field_dimensions: &FieldDimensions,
        pose_hint: Option<Isometry3<Robot, Field>>,
        parameters: &FieldMarkAssociationParameters,
        include_debug: bool,
    ) -> GlobalVisualLocalization {
        let localizer = GlobalAssociator::new(parameters.global_localizer);
        let input = GlobalLocalizationInput {
            visual_features,
            field_dimensions,
            ground_to_robot: camera_matrix.ground_to_robot,
            robot_to_camera: robot_to_camera(camera_matrix),
            camera_intrinsic: camera_matrix.intrinsics,
            pose_hint,
        };

        let pose_hint_result =
            localizer.associate_with_pose_hint(input.clone(), parameters.pose_hint);
        if !pose_hint_result.associations.is_empty() {
            if pose_hint_result.is_healthy(parameters.pose_hint) {
                self.has_global_lock = true;
                self.reset_recovery();
            }
            return GlobalVisualLocalization {
                debug: None,
                associations: pose_hint_field_mark_associations(&pose_hint_result),
            };
        }

        self.note_weak_pose_hint_frame();
        let result = localizer.localize(input);
        let debug = if include_debug {
            result.as_ref().map(global_localization_debug_from_result)
        } else {
            None
        };
        let associations =
            self.global_recovery_associations(result.as_ref(), pose_hint, parameters);

        GlobalVisualLocalization {
            debug,
            associations,
        }
    }

    fn note_weak_pose_hint_frame(&mut self) {
        self.weak_pose_hint_frames = self.weak_pose_hint_frames.saturating_add(1);
        self.has_global_lock = false;
    }

    fn reset_recovery(&mut self) {
        self.weak_pose_hint_frames = 0;
        self.pending_global_recovery = None;
    }

    fn global_recovery_associations(
        &mut self,
        result: Option<&GlobalLocalizationResult>,
        pose_hint: Option<Isometry3<Robot, Field>>,
        parameters: &FieldMarkAssociationParameters,
    ) -> Vec<FieldMarkAssociation> {
        if self.weak_pose_hint_frames < parameters.pose_hint.recovery_frames {
            return Vec::new();
        }
        let Some(result) = result else {
            self.pending_global_recovery = None;
            return Vec::new();
        };
        let Some(associations) = result.unique_feature_associations() else {
            self.pending_global_recovery = None;
            return Vec::new();
        };
        let robot_to_field = result.associations().robot_to_field;
        if let Some(pose_hint) = pose_hint {
            if !poses_agree(robot_to_field, pose_hint, parameters.pose_hint) {
                self.pending_global_recovery = None;
                return Vec::new();
            }
            self.has_global_lock = true;
            self.reset_recovery();
            return field_mark_associations(
                associations.iter(),
                FieldMarkAssociationKind::GlobalUnique,
            );
        }

        let associations =
            field_mark_associations(associations.iter(), FieldMarkAssociationKind::GlobalUnique);
        let consecutive_frames = self
            .pending_global_recovery
            .as_ref()
            .filter(|pending| {
                poses_agree(robot_to_field, pending.robot_to_field, parameters.pose_hint)
            })
            .map_or(1, |pending| pending.consecutive_frames + 1);
        self.pending_global_recovery = Some(GlobalRecoveryCandidate {
            robot_to_field,
            associations: associations.clone(),
            consecutive_frames,
        });
        if consecutive_frames >= parameters.pose_hint.recovery_frames {
            let associations = self
                .pending_global_recovery
                .take()
                .map(|pending| pending.associations)
                .unwrap_or_default();
            self.has_global_lock = true;
            self.weak_pose_hint_frames = 0;
            associations
        } else {
            Vec::new()
        }
    }
}

fn pose_hint_field_mark_associations(
    result: &PoseHintAssociationResult,
) -> Vec<FieldMarkAssociation> {
    field_mark_associations(
        result.associations.iter(),
        FieldMarkAssociationKind::PoseHint,
    )
}

fn poses_agree(
    left: Isometry3<Robot, Field>,
    right: Isometry3<Robot, Field>,
    config: PoseHintAssociationParameters,
) -> bool {
    let translation_delta = left.inner.translation.vector - right.inner.translation.vector;
    let translation_distance =
        nalgebra::Vector2::new(translation_delta.x, translation_delta.y).norm();
    let yaw_distance = yaw_difference(left, right).abs();
    translation_distance <= config.recovery_max_pose_distance
        && yaw_distance <= config.recovery_max_pose_angle
}

fn yaw_difference(left: Isometry3<Robot, Field>, right: Isometry3<Robot, Field>) -> f32 {
    let (_, _, left_yaw) = left.inner.rotation.euler_angles();
    let (_, _, right_yaw) = right.inner.rotation.euler_angles();
    let mut difference = left_yaw - right_yaw;
    while difference > std::f32::consts::PI {
        difference -= std::f32::consts::TAU;
    }
    while difference < -std::f32::consts::PI {
        difference += std::f32::consts::TAU;
    }
    difference
}

/// Runs global localization and returns debug data plus backend-safe associations.
pub fn localize_global_visual_features(
    visual_features: &DetectedVisualFeatures,
    camera_matrix: &CameraMatrix,
    field_dimensions: &FieldDimensions,
    pose_hint: Option<Isometry3<Robot, Field>>,
    parameters: &GlobalLocalizerParameters,
) -> GlobalVisualLocalization {
    localize_global_visual_features_with_debug(
        visual_features,
        camera_matrix,
        field_dimensions,
        pose_hint,
        parameters,
        true,
    )
}

fn localize_global_visual_features_with_debug(
    visual_features: &DetectedVisualFeatures,
    camera_matrix: &CameraMatrix,
    field_dimensions: &FieldDimensions,
    pose_hint: Option<Isometry3<Robot, Field>>,
    parameters: &GlobalLocalizerParameters,
    include_debug: bool,
) -> GlobalVisualLocalization {
    let localizer = GlobalAssociator::new(*parameters);
    let result = localizer.localize(GlobalLocalizationInput {
        visual_features,
        field_dimensions,
        ground_to_robot: camera_matrix.ground_to_robot,
        robot_to_camera: robot_to_camera(camera_matrix),
        camera_intrinsic: camera_matrix.intrinsics,
        pose_hint,
    });

    GlobalVisualLocalization {
        debug: if include_debug {
            result.as_ref().map(global_localization_debug_from_result)
        } else {
            None
        },
        associations: result
            .as_ref()
            .and_then(GlobalLocalizationResult::unique_feature_associations)
            .map(|associations| {
                field_mark_associations(associations.iter(), FieldMarkAssociationKind::GlobalUnique)
            })
            .unwrap_or_default(),
    }
}

fn field_mark_associations<'a>(
    associations: impl IntoIterator<Item = &'a FeatureAssociation>,
    kind: FieldMarkAssociationKind,
) -> Vec<FieldMarkAssociation> {
    associations
        .into_iter()
        .map(|association| FieldMarkAssociation {
            detection: association.detection,
            field_point: association.field_point.extend(0.0),
            kind,
        })
        .collect()
}

/// Runs global localization and returns per-feature debug data for visual inspection.
pub fn localize_global_visual_features_detailed_debug(
    visual_features: &DetectedVisualFeatures,
    camera_matrix: &CameraMatrix,
    field_dimensions: &FieldDimensions,
    pose_hint: Option<Isometry3<Robot, Field>>,
    parameters: &GlobalLocalizerParameters,
) -> Option<GlobalLocalizationDetailedDebug> {
    let localizer = GlobalAssociator::new(*parameters);
    localizer.localize_detailed(GlobalLocalizationInput {
        visual_features,
        field_dimensions,
        ground_to_robot: camera_matrix.ground_to_robot,
        robot_to_camera: robot_to_camera(camera_matrix),
        camera_intrinsic: camera_matrix.intrinsics,
        pose_hint,
    })
}

fn global_localization_debug_from_result(
    result: &GlobalLocalizationResult,
) -> GlobalLocalizationDebug {
    let status = match result {
        GlobalLocalizationResult::Ambiguous(_) => GlobalLocalizationDebugStatus::Ambiguous,
        GlobalLocalizationResult::UniqueModuloSymmetry(_) => {
            GlobalLocalizationDebugStatus::UniqueModuloSymmetry
        }
    };
    let associations = result.associations();
    GlobalLocalizationDebug {
        robot_to_field: associations.robot_to_field,
        status,
        inliers: associations.score.inliers,
        candidate_score: associations.score.candidate_score,
        metric_rms_residual: associations.score.metric_rms_residual,
        reprojection_rmse: associations.score.reprojection_rmse,
        total_cost: associations.score.total_cost,
    }
}

/// Extracts goalpost image points from object detections.
pub fn find_detected_goalposts(detections: &[Object<RobocupObjectLabel>]) -> Vec<Point2<Pixel>> {
    find_detected_visual_features(detections)
        .goalposts
        .into_iter()
        .map(|feature| feature.pixel)
        .collect()
}

/// Field-feature detection used by global localization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DetectedVisualFeature {
    /// Image point used for projection and association.
    pub pixel: Point2<Pixel>,
    /// Detector confidence in `[0, 1]`; invalid or low-confidence detections are ignored later.
    pub confidence: f32,
}

impl DetectedVisualFeature {
    fn new(pixel: Point2<Pixel>, confidence: f32) -> Self {
        Self { pixel, confidence }
    }
}

/// Field-feature detections grouped by the landmark class used by global localization.
#[derive(Debug, Default, PartialEq)]
pub struct DetectedVisualFeatures {
    /// Goalpost detections, represented by bottom-center image points.
    pub goalposts: Vec<DetectedVisualFeature>,
    /// L-crossing spot detections, represented by bounding-box centers.
    pub l_spots: Vec<DetectedVisualFeature>,
    /// T-crossing spot detections, represented by bounding-box centers.
    pub t_spots: Vec<DetectedVisualFeature>,
    /// Penalty spot detections, represented by bounding-box centers.
    pub penalty_spots: Vec<DetectedVisualFeature>,
}

impl DetectedVisualFeatures {
    /// Counts detections from classes supported by global localization.
    pub fn supported_feature_count(&self) -> usize {
        self.goalposts.len() + self.l_spots.len() + self.t_spots.len() + self.penalty_spots.len()
    }
}

/// Extracts all field-feature detections supported by global localization.
pub fn find_detected_visual_features(
    detections: &[Object<RobocupObjectLabel>],
) -> DetectedVisualFeatures {
    detections
        .iter()
        .fold(DetectedVisualFeatures::default(), |mut features, object| {
            let confidence = object.bounding_box.confidence;
            match object.label {
                RobocupObjectLabel::GoalPost => features.goalposts.push(
                    DetectedVisualFeature::new(pixel_bottom_center(object), confidence),
                ),
                RobocupObjectLabel::LSpot => features
                    .l_spots
                    .push(DetectedVisualFeature::new(pixel_center(object), confidence)),
                RobocupObjectLabel::TSpot => features
                    .t_spots
                    .push(DetectedVisualFeature::new(pixel_center(object), confidence)),
                RobocupObjectLabel::PenaltySpot => features
                    .penalty_spots
                    .push(DetectedVisualFeature::new(pixel_center(object), confidence)),
                _ => {}
            }
            features
        })
}

fn pixel_bottom_center(object: &Object<RobocupObjectLabel>) -> Point2<Pixel> {
    let area = object.bounding_box.area;
    point![(area.min.x() + area.max.x()) * 0.5, area.max.y()]
}

fn pixel_center(object: &Object<RobocupObjectLabel>) -> Point2<Pixel> {
    let area = object.bounding_box.area;
    point![
        (area.min.x() + area.max.x()) * 0.5,
        (area.min.y() + area.max.y()) * 0.5
    ]
}

fn robot_to_camera(camera_matrix: &CameraMatrix) -> Isometry3<Robot, Camera> {
    camera_matrix.head_to_camera * camera_matrix.robot_to_head
}

#[cfg(test)]
mod tests {
    use coordinate_systems::{Camera, Field, Head, Pixel, Robot};
    use geometry::rectangle::Rectangle;
    use linear_algebra::{IntoTransform as _, vector};
    use types::bounding_box::BoundingBox;
    use types::field_dimensions::{FieldDimensions, Half, Side};

    use super::*;

    #[test]
    fn goalpost_detection_uses_pixel_bottom_center() {
        let detections = vec![Object {
            label: RobocupObjectLabel::GoalPost,
            bounding_box: BoundingBox {
                area: Rectangle {
                    min: point![10.0, 20.0],
                    max: point![30.0, 50.0],
                },
                confidence: 1.0,
            },
        }];

        let goalposts = find_detected_goalposts(&detections);

        assert_eq!(goalposts.len(), 1);
        assert_eq!(goalposts[0], point![20.0, 50.0]);
    }

    #[test]
    fn spot_detections_use_pixel_center() {
        let detections = vec![
            Object {
                label: RobocupObjectLabel::LSpot,
                bounding_box: BoundingBox {
                    area: Rectangle {
                        min: point![10.0, 20.0],
                        max: point![30.0, 50.0],
                    },
                    confidence: 1.0,
                },
            },
            Object {
                label: RobocupObjectLabel::TSpot,
                bounding_box: BoundingBox {
                    area: Rectangle {
                        min: point![40.0, 60.0],
                        max: point![60.0, 80.0],
                    },
                    confidence: 1.0,
                },
            },
            Object {
                label: RobocupObjectLabel::PenaltySpot,
                bounding_box: BoundingBox {
                    area: Rectangle {
                        min: point![70.0, 90.0],
                        max: point![90.0, 110.0],
                    },
                    confidence: 1.0,
                },
            },
        ];

        let features = find_detected_visual_features(&detections);

        assert_eq!(feature_pixels(&features.l_spots), vec![point![20.0, 35.0]]);
        assert_eq!(feature_pixels(&features.t_spots), vec![point![50.0, 70.0]]);
        assert_eq!(
            feature_pixels(&features.penalty_spots),
            vec![point![80.0, 100.0]]
        );
        assert_eq!(
            features.l_spots.first().map(|feature| feature.confidence),
            Some(1.0)
        );
    }

    fn feature_pixels(features: &[DetectedVisualFeature]) -> Vec<Point2<Pixel>> {
        features.iter().map(|feature| feature.pixel).collect()
    }

    #[test]
    fn pose_hint_associations_are_preferred_when_reprojection_matches() {
        let field = FieldDimensions::SPL_2025;
        let pose = robot_to_field(0.0, 0.0, 0.0);
        let camera_matrix = synthetic_camera_matrix();
        let features = features_projected_from_pose(
            &camera_matrix,
            pose,
            [
                (
                    VisualFeatureClass::GoalPost,
                    field.goal_post(Half::Opponent, Side::Left),
                ),
                (
                    VisualFeatureClass::GoalPost,
                    field.goal_post(Half::Opponent, Side::Right),
                ),
                (
                    VisualFeatureClass::PenaltySpot,
                    field.penalty_spot(Half::Opponent),
                ),
                (VisualFeatureClass::TSpot, field.t_crossing(Side::Left)),
            ],
        );

        let localization = associate_visual_features(
            &features,
            &camera_matrix,
            &field,
            Some(pose),
            &FieldMarkAssociationParameters::default(),
        );

        assert!(!localization.associations.is_empty());
        assert!(
            localization
                .associations
                .iter()
                .all(|association| association.kind == FieldMarkAssociationKind::PoseHint),
            "pose-consistent reprojection associations should be used before global association: {:?}",
            localization.associations
        );
    }

    #[test]
    fn global_association_is_used_when_pose_hint_reprojection_gate_rejects_features() {
        let field = FieldDimensions::SPL_2025;
        let true_pose = robot_to_field(0.0, 0.0, 0.0);
        let wrong_pose_hint = robot_to_field(0.6, 0.0, 0.0);
        let camera_matrix = synthetic_camera_matrix();
        let features = features_projected_from_pose(
            &camera_matrix,
            true_pose,
            [
                (
                    VisualFeatureClass::GoalPost,
                    field.goal_post(Half::Opponent, Side::Left),
                ),
                (
                    VisualFeatureClass::GoalPost,
                    field.goal_post(Half::Opponent, Side::Right),
                ),
                (
                    VisualFeatureClass::PenaltySpot,
                    field.penalty_spot(Half::Opponent),
                ),
                (VisualFeatureClass::TSpot, field.t_crossing(Side::Left)),
            ],
        );

        let mut parameters = FieldMarkAssociationParameters::default();
        parameters.global_localizer.min_inliers = 3;
        parameters.pose_hint.max_reprojection_error_px = 1.0;

        let localization = associate_visual_features(
            &features,
            &camera_matrix,
            &field,
            Some(wrong_pose_hint),
            &parameters,
        );

        assert!(!localization.associations.is_empty());
        assert!(
            localization
                .associations
                .iter()
                .all(|association| association.kind == FieldMarkAssociationKind::GlobalUnique),
            "global association should be used after pose-hint reprojection is rejected: {:?}",
            localization.associations
        );
    }

    #[test]
    fn fewer_than_three_features_still_use_pose_hint_reprojection() {
        let field = FieldDimensions::SPL_2025;
        let pose = robot_to_field(0.0, 0.0, 0.0);
        let camera_matrix = synthetic_camera_matrix();
        let features = features_projected_from_pose(
            &camera_matrix,
            pose,
            [(
                VisualFeatureClass::PenaltySpot,
                field.penalty_spot(Half::Opponent),
            )],
        );

        let localization = associate_visual_features(
            &features,
            &camera_matrix,
            &field,
            Some(pose),
            &FieldMarkAssociationParameters::default(),
        );

        assert_eq!(localization.associations.len(), 1);
        assert_eq!(
            localization.associations[0].kind,
            FieldMarkAssociationKind::PoseHint
        );
    }

    fn synthetic_camera_matrix() -> CameraMatrix {
        CameraMatrix::from_normalized_focal_and_center(
            nalgebra::vector![60.0 / 1280.0, 60.0 / 960.0],
            nalgebra::point![0.5, 0.5],
            vector![<Pixel>, 1280.0, 960.0],
            Isometry3::identity(),
            Isometry3::identity(),
            nalgebra::Isometry3::translation(0.0, 0.0, 0.5).framed_transform::<Head, Camera>(),
        )
    }

    fn robot_to_field(x: f32, y: f32, yaw: f32) -> Isometry3<Robot, Field> {
        Isometry3::<Robot, Field>::wrap(nalgebra::Isometry3::from_parts(
            nalgebra::Translation3::new(x, y, 0.0),
            nalgebra::UnitQuaternion::from_axis_angle(&nalgebra::Vector3::z_axis(), yaw),
        ))
    }

    fn features_projected_from_pose<const N: usize>(
        camera_matrix: &CameraMatrix,
        robot_to_field: Isometry3<Robot, Field>,
        landmarks: [(VisualFeatureClass, Point2<Field>); N],
    ) -> DetectedVisualFeatures {
        let field_to_camera = robot_to_camera(camera_matrix) * robot_to_field.inverse();
        let mut features = DetectedVisualFeatures::default();

        for (class, field_point) in landmarks {
            let camera_point = field_to_camera * field_point.extend(0.0);
            let pixel = camera_matrix.intrinsics.project(camera_point.coords());
            let feature = DetectedVisualFeature {
                pixel,
                confidence: 0.9,
            };

            match class {
                VisualFeatureClass::GoalPost => features.goalposts.push(feature),
                VisualFeatureClass::LSpot => features.l_spots.push(feature),
                VisualFeatureClass::TSpot => features.t_spots.push(feature),
                VisualFeatureClass::PenaltySpot => features.penalty_spots.push(feature),
            }
        }

        features
    }
}
