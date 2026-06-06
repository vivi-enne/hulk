use linear_algebra::nalgebra::{
    DMatrix, Isometry3, Matrix3, Point3, Rotation3, SMatrix, SVector, Translation3, UnitQuaternion,
    Vector2, Vector3, Vector4,
};
use ndarray::Array2;
use thiserror::Error;

use crate::features::XFeatOutput;

type ProjectionMatrix = SMatrix<f32, 3, 4>;
type Matrix6 = SMatrix<f32, 6, 6>;
type Vector6 = SVector<f32, 6>;

const MINIMUM_COSINE_SIMILARITY: f32 = 0.82;
const TRIANGULATION_EPSILON: f32 = 1e-6;
const MINIMUM_PNP_CORRESPONDENCES: usize = 6;
const PNP_RANSAC_ITERATIONS: usize = 128;
const PNP_RANSAC_REPROJECTION_THRESHOLD: f32 = 6.0;
const POSE_REFINEMENT_ITERATIONS: usize = 10;

#[derive(Debug, Error)]
pub enum Matcher3DError {
    #[error("{name} calibration has shape {actual:?}, expected [3, 4]")]
    InvalidCalibrationShape {
        name: &'static str,
        actual: [usize; 2],
    },
    #[error("left calibration intrinsics are singular")]
    SingularLeftIntrinsics,
    #[error("{field} has shape {actual:?}, expected {expected}")]
    UnexpectedFeatureShape {
        field: &'static str,
        actual: Vec<usize>,
        expected: &'static str,
    },
    #[error("descriptor dimensions differ ({left} vs {right})")]
    DescriptorDimensionMismatch { left: usize, right: usize },
}

pub struct Matcher3D {
    left_projection: ProjectionMatrix,
    right_projection: ProjectionMatrix,
    left_intrinsics: Matrix3<f32>,
    left_intrinsics_inverse: Matrix3<f32>,
    previous_features: Option<ExtractedFeatures>,
    previous_points: Vec<Point3<f32>>,
    candidate_tracker: LandmarkCandidateTracker,
}

#[derive(Debug, Clone)]
pub struct MatcherOutput {
    pub isometry: Isometry3<f32>,
    pub proposed_points: Array2<f32>,
    pub proposed_descriptors: Array2<f32>,
}

impl Matcher3D {
    pub fn initialize(
        left_calibration: Array2<f32>,
        right_calibration: Array2<f32>,
    ) -> Result<Self, Matcher3DError> {
        let left_projection = projection_from_array(left_calibration, "left")?;
        let right_projection = projection_from_array(right_calibration, "right")?;
        let left_intrinsics = left_projection.fixed_view::<3, 3>(0, 0).into_owned();
        let left_intrinsics_inverse = left_intrinsics
            .try_inverse()
            .ok_or(Matcher3DError::SingularLeftIntrinsics)?;

        Ok(Self {
            left_projection,
            right_projection,
            left_intrinsics,
            left_intrinsics_inverse,
            previous_features: None,
            previous_points: Vec::new(),
            candidate_tracker: LandmarkCandidateTracker::default(),
        })
    }

    pub fn step(&mut self, features: XFeatOutput) -> Result<MatcherOutput, Matcher3DError> {
        let (left, right) = xfeat_to_features(&features)?;
        let (left_matches, right_matches) = match_descriptors(&left, &right)?;
        let left_filtered = left.select_indices(&left_matches);
        let right_filtered = right.select_indices(&right_matches);

        let triangulation = triangulate_points(
            &left_filtered,
            &right_filtered,
            &self.left_projection,
            &self.right_projection,
        );
        let left_non_singular = left_filtered.select_mask(&triangulation.non_singular_mask);

        let Some(previous_features) = &self.previous_features else {
            self.previous_features = Some(left_non_singular);
            self.previous_points = triangulation.points;
            return Ok(MatcherOutput::empty(left.descriptor_dimensions));
        };

        let (previous_matches, current_matches) =
            match_descriptors(previous_features, &left_non_singular)?;

        let matched_points = previous_matches
            .iter()
            .map(|&index| self.previous_points[index])
            .collect::<Vec<_>>();
        let image_positions = current_matches
            .iter()
            .map(|&index| left_non_singular.keypoints[index])
            .collect::<Vec<_>>();

        let isometry = solve_pose_transform(
            &matched_points,
            &image_positions,
            &self.left_intrinsics,
            &self.left_intrinsics_inverse,
        )
        .unwrap_or_else(Isometry3::identity);

        let (proposed_points, proposed_descriptors) = self.candidate_tracker.update(
            &triangulation.points,
            &left_non_singular,
            &previous_matches,
            &current_matches,
        );

        self.previous_points = triangulation.points;
        self.previous_features = Some(left_non_singular);

        Ok(MatcherOutput {
            isometry,
            proposed_points,
            proposed_descriptors,
        })
    }
}

impl MatcherOutput {
    fn empty(descriptor_dimensions: usize) -> Self {
        Self {
            isometry: Isometry3::identity(),
            proposed_points: Array2::zeros((0, 3)),
            proposed_descriptors: Array2::zeros((0, descriptor_dimensions)),
        }
    }
}

#[derive(Debug, Clone)]
struct ExtractedFeatures {
    keypoints: Vec<Vector2<f32>>,
    scores: Vec<f32>,
    descriptors: Vec<f32>,
    descriptor_dimensions: usize,
}

impl ExtractedFeatures {
    fn len(&self) -> usize {
        self.keypoints.len()
    }

    fn descriptor(&self, index: usize) -> &[f32] {
        let start = index * self.descriptor_dimensions;
        let end = start + self.descriptor_dimensions;
        &self.descriptors[start..end]
    }

    fn select_indices(&self, indices: &[usize]) -> Self {
        let mut keypoints = Vec::with_capacity(indices.len());
        let mut scores = Vec::with_capacity(indices.len());
        let mut descriptors = Vec::with_capacity(indices.len() * self.descriptor_dimensions);

        for &index in indices {
            keypoints.push(self.keypoints[index]);
            scores.push(self.scores[index]);
            descriptors.extend_from_slice(self.descriptor(index));
        }

        Self {
            keypoints,
            scores,
            descriptors,
            descriptor_dimensions: self.descriptor_dimensions,
        }
    }

    fn select_mask(&self, mask: &[bool]) -> Self {
        debug_assert_eq!(self.len(), mask.len());
        let indices = mask
            .iter()
            .enumerate()
            .filter_map(|(index, is_selected)| is_selected.then_some(index))
            .collect::<Vec<_>>();
        self.select_indices(&indices)
    }
}

#[derive(Debug)]
struct TriangulationOutput {
    points: Vec<Point3<f32>>,
    non_singular_mask: Vec<bool>,
}

#[derive(Debug)]
struct LandmarkCandidateTracker {
    minimum_consecutive_frames: u32,
    feature_track_counts: Option<Vec<u32>>,
}

impl Default for LandmarkCandidateTracker {
    fn default() -> Self {
        Self {
            minimum_consecutive_frames: 3,
            feature_track_counts: None,
        }
    }
}

impl LandmarkCandidateTracker {
    fn update(
        &mut self,
        current_points: &[Point3<f32>],
        current_features: &ExtractedFeatures,
        previous_match_indices: &[usize],
        current_match_indices: &[usize],
    ) -> (Array2<f32>, Array2<f32>) {
        let Some(previous_track_counts) = &self.feature_track_counts else {
            self.feature_track_counts = Some(vec![1; current_features.len()]);
            return (
                Array2::zeros((0, 3)),
                Array2::zeros((0, current_features.descriptor_dimensions)),
            );
        };

        let mut current_track_counts = vec![1; current_features.len()];
        for (&previous_index, &current_index) in
            previous_match_indices.iter().zip(current_match_indices)
        {
            if let Some(previous_count) = previous_track_counts.get(previous_index) {
                current_track_counts[current_index] = previous_count + 1;
            }
        }

        let mature_indices = current_track_counts
            .iter()
            .enumerate()
            .filter_map(|(index, &count)| {
                (count == self.minimum_consecutive_frames).then_some(index)
            })
            .collect::<Vec<_>>();

        let proposed_points = points_to_array(
            &mature_indices
                .iter()
                .map(|&index| current_points[index])
                .collect::<Vec<_>>(),
        );
        let proposed_descriptors = descriptors_to_array(current_features, &mature_indices);

        self.feature_track_counts = Some(current_track_counts);

        (proposed_points, proposed_descriptors)
    }
}

fn projection_from_array(
    calibration: Array2<f32>,
    name: &'static str,
) -> Result<ProjectionMatrix, Matcher3DError> {
    let (rows, columns) = calibration.dim();
    if rows != 3 || columns != 4 {
        return Err(Matcher3DError::InvalidCalibrationShape {
            name,
            actual: [rows, columns],
        });
    }

    let mut projection = ProjectionMatrix::zeros();
    for row in 0..3 {
        for column in 0..4 {
            projection[(row, column)] = calibration[(row, column)];
        }
    }
    Ok(projection)
}

fn xfeat_to_features(
    features: &XFeatOutput,
) -> Result<(ExtractedFeatures, ExtractedFeatures), Matcher3DError> {
    let (keypoint_views, feature_count, keypoint_dimensions) = features.keypoints.dim();
    if keypoint_views != 2 || keypoint_dimensions != 2 {
        return Err(Matcher3DError::UnexpectedFeatureShape {
            field: "keypoints",
            actual: vec![keypoint_views, feature_count, keypoint_dimensions],
            expected: "[2, N, 2]",
        });
    }

    let (score_views, score_count) = features.scores.dim();
    if score_views != 2 || score_count != feature_count {
        return Err(Matcher3DError::UnexpectedFeatureShape {
            field: "scores",
            actual: vec![score_views, score_count],
            expected: "[2, N]",
        });
    }

    let (descriptor_views, descriptor_count, descriptor_dimensions) = features.descriptors.dim();
    if descriptor_views != 2 || descriptor_count != feature_count {
        return Err(Matcher3DError::UnexpectedFeatureShape {
            field: "descriptors",
            actual: vec![descriptor_views, descriptor_count, descriptor_dimensions],
            expected: "[2, N, D]",
        });
    }

    Ok((
        extract_view_features(features, 0, feature_count, descriptor_dimensions),
        extract_view_features(features, 1, feature_count, descriptor_dimensions),
    ))
}

fn extract_view_features(
    features: &XFeatOutput,
    view: usize,
    feature_count: usize,
    descriptor_dimensions: usize,
) -> ExtractedFeatures {
    let mut keypoints = Vec::with_capacity(feature_count);
    let mut scores = Vec::with_capacity(feature_count);
    let mut descriptors = Vec::with_capacity(feature_count * descriptor_dimensions);

    for feature_index in 0..feature_count {
        keypoints.push(Vector2::new(
            features.keypoints[(view, feature_index, 0)] as f32,
            features.keypoints[(view, feature_index, 1)] as f32,
        ));
        scores.push(features.scores[(view, feature_index)]);
        for dimension in 0..descriptor_dimensions {
            descriptors.push(features.descriptors[(view, feature_index, dimension)]);
        }
    }

    ExtractedFeatures {
        keypoints,
        scores,
        descriptors,
        descriptor_dimensions,
    }
}

fn match_descriptors(
    descriptors_a: &ExtractedFeatures,
    descriptors_b: &ExtractedFeatures,
) -> Result<(Vec<usize>, Vec<usize>), Matcher3DError> {
    if descriptors_a.descriptor_dimensions != descriptors_b.descriptor_dimensions {
        return Err(Matcher3DError::DescriptorDimensionMismatch {
            left: descriptors_a.descriptor_dimensions,
            right: descriptors_b.descriptor_dimensions,
        });
    }

    if descriptors_a.len() == 0 || descriptors_b.len() == 0 {
        return Ok((Vec::new(), Vec::new()));
    }

    let mut forward_indices = vec![0; descriptors_a.len()];
    let mut forward_similarities = vec![f32::NEG_INFINITY; descriptors_a.len()];
    for index_a in 0..descriptors_a.len() {
        for index_b in 0..descriptors_b.len() {
            let similarity = dot_product(
                descriptors_a.descriptor(index_a),
                descriptors_b.descriptor(index_b),
            );
            if similarity > forward_similarities[index_a] {
                forward_similarities[index_a] = similarity;
                forward_indices[index_a] = index_b;
            }
        }
    }

    let mut backward_indices = vec![0; descriptors_b.len()];
    let mut backward_similarities = vec![f32::NEG_INFINITY; descriptors_b.len()];
    for index_b in 0..descriptors_b.len() {
        for index_a in 0..descriptors_a.len() {
            let similarity = dot_product(
                descriptors_b.descriptor(index_b),
                descriptors_a.descriptor(index_a),
            );
            if similarity > backward_similarities[index_b] {
                backward_similarities[index_b] = similarity;
                backward_indices[index_b] = index_a;
            }
        }
    }

    let mut matched_a = Vec::new();
    let mut matched_b = Vec::new();
    for (index_a, &index_b) in forward_indices.iter().enumerate() {
        if backward_indices[index_b] == index_a
            && forward_similarities[index_a] > MINIMUM_COSINE_SIMILARITY
        {
            matched_a.push(index_a);
            matched_b.push(index_b);
        }
    }

    Ok((matched_a, matched_b))
}

fn dot_product(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

fn triangulate_points(
    left: &ExtractedFeatures,
    right: &ExtractedFeatures,
    left_projection: &ProjectionMatrix,
    right_projection: &ProjectionMatrix,
) -> TriangulationOutput {
    debug_assert_eq!(left.len(), right.len());

    let mut points = Vec::new();
    let mut non_singular_mask = Vec::with_capacity(left.len());
    for (&left_keypoint, &right_keypoint) in left.keypoints.iter().zip(&right.keypoints) {
        let point = triangulate_point(
            left_keypoint,
            right_keypoint,
            left_projection,
            right_projection,
        );
        non_singular_mask.push(point.is_some());
        if let Some(point) = point {
            points.push(point);
        }
    }

    TriangulationOutput {
        points,
        non_singular_mask,
    }
}

fn triangulate_point(
    left_keypoint: Vector2<f32>,
    right_keypoint: Vector2<f32>,
    left_projection: &ProjectionMatrix,
    right_projection: &ProjectionMatrix,
) -> Option<Point3<f32>> {
    let mut equations = SMatrix::<f32, 4, 4>::zeros();
    for column in 0..4 {
        equations[(0, column)] =
            left_keypoint.x * left_projection[(2, column)] - left_projection[(0, column)];
        equations[(1, column)] =
            left_keypoint.y * left_projection[(2, column)] - left_projection[(1, column)];
        equations[(2, column)] =
            right_keypoint.x * right_projection[(2, column)] - right_projection[(0, column)];
        equations[(3, column)] =
            right_keypoint.y * right_projection[(2, column)] - right_projection[(1, column)];
    }

    let singular_value_decomposition = equations.svd(true, true);
    let v_transpose = singular_value_decomposition.v_t?;
    let homogeneous = Vector4::new(
        v_transpose[(3, 0)],
        v_transpose[(3, 1)],
        v_transpose[(3, 2)],
        v_transpose[(3, 3)],
    );

    if homogeneous.w.abs() <= TRIANGULATION_EPSILON
        || !homogeneous.iter().all(|value| value.is_finite())
    {
        return None;
    }

    Some(Point3::new(
        homogeneous.x / homogeneous.w,
        homogeneous.y / homogeneous.w,
        homogeneous.z / homogeneous.w,
    ))
}

fn solve_pose_transform(
    previous_points: &[Point3<f32>],
    current_image_points: &[Vector2<f32>],
    intrinsics: &Matrix3<f32>,
    intrinsics_inverse: &Matrix3<f32>,
) -> Option<Isometry3<f32>> {
    if previous_points.len() < MINIMUM_PNP_CORRESPONDENCES
        || current_image_points.len() < MINIMUM_PNP_CORRESPONDENCES
    {
        return None;
    }
    debug_assert_eq!(previous_points.len(), current_image_points.len());

    let all_indices = (0..previous_points.len()).collect::<Vec<_>>();
    let mut best_pose = estimate_pose_dlt(
        previous_points,
        current_image_points,
        &all_indices,
        intrinsics_inverse,
    )?;
    let mut best_score = score_pose(
        &best_pose,
        previous_points,
        current_image_points,
        intrinsics,
        PNP_RANSAC_REPROJECTION_THRESHOLD,
    );

    for iteration in 0..PNP_RANSAC_ITERATIONS {
        let sample = sample_indices(previous_points.len(), iteration);
        let Some(candidate_pose) = estimate_pose_dlt(
            previous_points,
            current_image_points,
            &sample,
            intrinsics_inverse,
        ) else {
            continue;
        };
        let candidate_score = score_pose(
            &candidate_pose,
            previous_points,
            current_image_points,
            intrinsics,
            PNP_RANSAC_REPROJECTION_THRESHOLD,
        );
        if candidate_score.is_better_than(best_score) {
            best_pose = candidate_pose;
            best_score = candidate_score;
        }
    }

    let inlier_indices = collect_inliers(
        &best_pose,
        previous_points,
        current_image_points,
        intrinsics,
        PNP_RANSAC_REPROJECTION_THRESHOLD,
    );
    if inlier_indices.len() >= MINIMUM_PNP_CORRESPONDENCES {
        best_pose = estimate_pose_dlt(
            previous_points,
            current_image_points,
            &inlier_indices,
            intrinsics_inverse,
        )
        .unwrap_or(best_pose);
    } else {
        return None;
    }

    Some(refine_pose(
        best_pose,
        previous_points,
        current_image_points,
        &inlier_indices,
        intrinsics,
    ))
}

fn estimate_pose_dlt(
    object_points: &[Point3<f32>],
    image_points: &[Vector2<f32>],
    indices: &[usize],
    intrinsics_inverse: &Matrix3<f32>,
) -> Option<Isometry3<f32>> {
    if indices.len() < MINIMUM_PNP_CORRESPONDENCES {
        return None;
    }

    let mut equations = DMatrix::<f32>::zeros(indices.len() * 2, 12);
    for (row_index, &point_index) in indices.iter().enumerate() {
        let point = object_points[point_index];
        let image_point = image_points[point_index];
        let normalized = intrinsics_inverse * Vector3::new(image_point.x, image_point.y, 1.0);
        if normalized.z.abs() <= TRIANGULATION_EPSILON {
            return None;
        }
        let normalized_x = normalized.x / normalized.z;
        let normalized_y = normalized.y / normalized.z;
        let homogeneous_point = [point.x, point.y, point.z, 1.0];

        for column in 0..4 {
            equations[(row_index * 2, column)] = homogeneous_point[column];
            equations[(row_index * 2, 8 + column)] = -normalized_x * homogeneous_point[column];
            equations[(row_index * 2 + 1, 4 + column)] = homogeneous_point[column];
            equations[(row_index * 2 + 1, 8 + column)] = -normalized_y * homogeneous_point[column];
        }
    }

    let singular_value_decomposition = equations.svd(true, true);
    let v_transpose = singular_value_decomposition.v_t?;
    let last_row = v_transpose.nrows() - 1;
    let mut pose = SMatrix::<f32, 3, 4>::zeros();
    for column in 0..4 {
        pose[(0, column)] = v_transpose[(last_row, column)];
        pose[(1, column)] = v_transpose[(last_row, 4 + column)];
        pose[(2, column)] = v_transpose[(last_row, 8 + column)];
    }

    pose_from_matrix(pose)
}

fn pose_from_matrix(mut pose: SMatrix<f32, 3, 4>) -> Option<Isometry3<f32>> {
    let mut rotation_scale = pose.fixed_view::<3, 3>(0, 0).into_owned();
    if rotation_scale.determinant() < 0.0 {
        pose = -pose;
        rotation_scale = -rotation_scale;
    }

    let translation_scale = Vector3::new(pose[(0, 3)], pose[(1, 3)], pose[(2, 3)]);
    let singular_value_decomposition = rotation_scale.svd(true, true);
    let u = singular_value_decomposition.u?;
    let v_transpose = singular_value_decomposition.v_t?;
    let singular_values = singular_value_decomposition.singular_values;
    let scale = singular_values.iter().sum::<f32>() / 3.0;
    if scale <= TRIANGULATION_EPSILON || !scale.is_finite() {
        return None;
    }

    let mut handedness = Matrix3::identity();
    if (u * v_transpose).determinant() < 0.0 {
        handedness[(2, 2)] = -1.0;
    }
    let rotation_matrix = u * handedness * v_transpose;
    if !rotation_matrix.iter().all(|value| value.is_finite()) {
        return None;
    }

    let rotation =
        UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(rotation_matrix));
    let translation = translation_scale / scale;
    if !translation.iter().all(|value| value.is_finite()) {
        return None;
    }

    Some(Isometry3::from_parts(
        Translation3::from(translation),
        rotation,
    ))
}

#[derive(Debug, Clone, Copy)]
struct PoseScore {
    inliers: usize,
    total_squared_error: f32,
}

impl PoseScore {
    fn is_better_than(self, other: Self) -> bool {
        self.inliers > other.inliers
            || (self.inliers == other.inliers
                && self.total_squared_error < other.total_squared_error)
    }
}

fn score_pose(
    pose: &Isometry3<f32>,
    object_points: &[Point3<f32>],
    image_points: &[Vector2<f32>],
    intrinsics: &Matrix3<f32>,
    threshold: f32,
) -> PoseScore {
    let threshold_squared = threshold * threshold;
    let mut inliers = 0;
    let mut total_squared_error = 0.0;

    for (object_point, image_point) in object_points.iter().zip(image_points) {
        let Some(error) = squared_reprojection_error(pose, object_point, image_point, intrinsics)
        else {
            continue;
        };
        if error <= threshold_squared {
            inliers += 1;
            total_squared_error += error;
        }
    }

    PoseScore {
        inliers,
        total_squared_error,
    }
}

fn collect_inliers(
    pose: &Isometry3<f32>,
    object_points: &[Point3<f32>],
    image_points: &[Vector2<f32>],
    intrinsics: &Matrix3<f32>,
    threshold: f32,
) -> Vec<usize> {
    let threshold_squared = threshold * threshold;
    object_points
        .iter()
        .zip(image_points)
        .enumerate()
        .filter_map(|(index, (object_point, image_point))| {
            let error = squared_reprojection_error(pose, object_point, image_point, intrinsics)?;
            (error <= threshold_squared).then_some(index)
        })
        .collect()
}

fn squared_reprojection_error(
    pose: &Isometry3<f32>,
    object_point: &Point3<f32>,
    image_point: &Vector2<f32>,
    intrinsics: &Matrix3<f32>,
) -> Option<f32> {
    let projection = project_point(pose, object_point, intrinsics)?;
    Some((projection - image_point).norm_squared())
}

fn project_point(
    pose: &Isometry3<f32>,
    object_point: &Point3<f32>,
    intrinsics: &Matrix3<f32>,
) -> Option<Vector2<f32>> {
    let camera_point = pose.transform_point(object_point);
    if camera_point.z <= TRIANGULATION_EPSILON {
        return None;
    }

    let projected = intrinsics * camera_point.coords;
    if projected.z.abs() <= TRIANGULATION_EPSILON
        || !projected.iter().all(|value| value.is_finite())
    {
        return None;
    }

    Some(Vector2::new(
        projected.x / projected.z,
        projected.y / projected.z,
    ))
}

fn refine_pose(
    mut pose: Isometry3<f32>,
    object_points: &[Point3<f32>],
    image_points: &[Vector2<f32>],
    indices: &[usize],
    intrinsics: &Matrix3<f32>,
) -> Isometry3<f32> {
    if indices.len() < MINIMUM_PNP_CORRESPONDENCES {
        return pose;
    }

    let mut current_error =
        total_reprojection_error(&pose, object_points, image_points, indices, intrinsics);
    for _ in 0..POSE_REFINEMENT_ITERATIONS {
        let Some(delta) =
            pose_refinement_delta(&pose, object_points, image_points, indices, intrinsics)
        else {
            break;
        };
        if delta.norm() < 1e-5 {
            break;
        }

        let candidate = pose_increment(delta) * pose;
        let candidate_error =
            total_reprojection_error(&candidate, object_points, image_points, indices, intrinsics);
        if candidate_error.is_finite() && candidate_error < current_error {
            pose = candidate;
            current_error = candidate_error;
        } else {
            break;
        }
    }

    pose
}

fn pose_refinement_delta(
    pose: &Isometry3<f32>,
    object_points: &[Point3<f32>],
    image_points: &[Vector2<f32>],
    indices: &[usize],
    intrinsics: &Matrix3<f32>,
) -> Option<Vector6> {
    let mut normal_matrix = Matrix6::zeros();
    let mut gradient = Vector6::zeros();
    let mut residual_count = 0;

    for &index in indices {
        let Some((projection, jacobian)) =
            projection_jacobian(pose, &object_points[index], intrinsics)
        else {
            continue;
        };
        let residual = projection - image_points[index];
        normal_matrix += jacobian.transpose() * jacobian;
        gradient += jacobian.transpose() * residual;
        residual_count += 1;
    }

    if residual_count < MINIMUM_PNP_CORRESPONDENCES {
        return None;
    }

    let damping = (normal_matrix.trace() / 6.0).max(1.0) * 1e-6;
    for index in 0..6 {
        normal_matrix[(index, index)] += damping;
    }

    normal_matrix.lu().solve(&(-gradient))
}

fn projection_jacobian(
    pose: &Isometry3<f32>,
    object_point: &Point3<f32>,
    intrinsics: &Matrix3<f32>,
) -> Option<(Vector2<f32>, SMatrix<f32, 2, 6>)> {
    let camera_point = pose.transform_point(object_point);
    if camera_point.z <= TRIANGULATION_EPSILON {
        return None;
    }

    let projected_homogeneous = intrinsics * camera_point.coords;
    if projected_homogeneous.z.abs() <= TRIANGULATION_EPSILON {
        return None;
    }

    let projection = Vector2::new(
        projected_homogeneous.x / projected_homogeneous.z,
        projected_homogeneous.y / projected_homogeneous.z,
    );

    let k0 = Vector3::new(intrinsics[(0, 0)], intrinsics[(0, 1)], intrinsics[(0, 2)]);
    let k1 = Vector3::new(intrinsics[(1, 0)], intrinsics[(1, 1)], intrinsics[(1, 2)]);
    let k2 = Vector3::new(intrinsics[(2, 0)], intrinsics[(2, 1)], intrinsics[(2, 2)]);
    let denominator = projected_homogeneous.z * projected_homogeneous.z;
    let d_u_d_camera = (k0 * projected_homogeneous.z - k2 * projected_homogeneous.x) / denominator;
    let d_v_d_camera = (k1 * projected_homogeneous.z - k2 * projected_homogeneous.y) / denominator;

    let x = camera_point.x;
    let y = camera_point.y;
    let z = camera_point.z;
    let motion_derivatives = [
        Vector3::new(1.0, 0.0, 0.0),
        Vector3::new(0.0, 1.0, 0.0),
        Vector3::new(0.0, 0.0, 1.0),
        Vector3::new(0.0, -z, y),
        Vector3::new(z, 0.0, -x),
        Vector3::new(-y, x, 0.0),
    ];

    let mut jacobian = SMatrix::<f32, 2, 6>::zeros();
    for column in 0..6 {
        jacobian[(0, column)] = d_u_d_camera.dot(&motion_derivatives[column]);
        jacobian[(1, column)] = d_v_d_camera.dot(&motion_derivatives[column]);
    }

    Some((projection, jacobian))
}

fn pose_increment(delta: Vector6) -> Isometry3<f32> {
    Isometry3::from_parts(
        Translation3::new(delta[0], delta[1], delta[2]),
        UnitQuaternion::from_scaled_axis(Vector3::new(delta[3], delta[4], delta[5])),
    )
}

fn total_reprojection_error(
    pose: &Isometry3<f32>,
    object_points: &[Point3<f32>],
    image_points: &[Vector2<f32>],
    indices: &[usize],
    intrinsics: &Matrix3<f32>,
) -> f32 {
    indices
        .iter()
        .map(|&index| {
            squared_reprojection_error(
                pose,
                &object_points[index],
                &image_points[index],
                intrinsics,
            )
            .unwrap_or(f32::INFINITY)
        })
        .sum()
}

fn sample_indices(total: usize, iteration: usize) -> [usize; MINIMUM_PNP_CORRESPONDENCES] {
    debug_assert!(total >= MINIMUM_PNP_CORRESPONDENCES);

    let mut state = (iteration as u64)
        .wrapping_add(1)
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ (total as u64);
    let mut indices = [usize::MAX; MINIMUM_PNP_CORRESPONDENCES];
    for output_index in 0..MINIMUM_PNP_CORRESPONDENCES {
        loop {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let candidate = (state as usize) % total;
            if !indices[..output_index].contains(&candidate) {
                indices[output_index] = candidate;
                break;
            }
        }
    }

    indices
}

fn points_to_array(points: &[Point3<f32>]) -> Array2<f32> {
    Array2::from_shape_fn((points.len(), 3), |(row, column)| points[row][column])
}

fn descriptors_to_array(features: &ExtractedFeatures, indices: &[usize]) -> Array2<f32> {
    Array2::from_shape_fn(
        (indices.len(), features.descriptor_dimensions),
        |(row, column)| features.descriptor(indices[row])[column],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn features_from_descriptors(descriptors: &[[f32; 3]]) -> ExtractedFeatures {
        ExtractedFeatures {
            keypoints: vec![Vector2::zeros(); descriptors.len()],
            scores: vec![1.0; descriptors.len()],
            descriptors: descriptors.iter().flatten().copied().collect(),
            descriptor_dimensions: 3,
        }
    }

    #[test]
    fn matches_mutual_nearest_neighbors() {
        let left = features_from_descriptors(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
        let right = features_from_descriptors(&[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.6, 0.6, 0.0]]);

        let (left_matches, right_matches) = match_descriptors(&left, &right).unwrap();

        assert_eq!(left_matches, vec![0, 1]);
        assert_eq!(right_matches, vec![1, 0]);
    }

    #[test]
    fn triangulates_stereo_correspondence() {
        let left_projection = ProjectionMatrix::from_row_slice(&[
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        ]);
        let right_projection = ProjectionMatrix::from_row_slice(&[
            1.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        ]);
        let point = triangulate_point(
            Vector2::new(0.5, 0.25),
            Vector2::new(0.25, 0.25),
            &left_projection,
            &right_projection,
        )
        .unwrap();

        assert!((point.x - 2.0).abs() < 1e-4);
        assert!((point.y - 1.0).abs() < 1e-4);
        assert!((point.z - 4.0).abs() < 1e-4);
    }

    #[test]
    fn estimates_pose_from_synthetic_correspondences() {
        let intrinsics = Matrix3::new(400.0, 0.0, 320.0, 0.0, 400.0, 240.0, 0.0, 0.0, 1.0);
        let intrinsics_inverse = intrinsics.try_inverse().unwrap();
        let expected_pose = Isometry3::from_parts(
            Translation3::new(0.2, -0.1, 0.4),
            UnitQuaternion::from_scaled_axis(Vector3::new(0.03, -0.02, 0.01)),
        );
        let object_points = vec![
            Point3::new(-1.0, -0.5, 4.0),
            Point3::new(0.8, -0.4, 5.0),
            Point3::new(-0.3, 0.7, 4.5),
            Point3::new(1.2, 0.3, 6.0),
            Point3::new(-0.8, 1.0, 5.5),
            Point3::new(0.4, -1.0, 4.8),
            Point3::new(1.5, 1.2, 7.0),
            Point3::new(-1.4, 0.2, 6.5),
        ];
        let image_points = object_points
            .iter()
            .map(|point| project_point(&expected_pose, point, &intrinsics).unwrap())
            .collect::<Vec<_>>();

        let pose = solve_pose_transform(
            &object_points,
            &image_points,
            &intrinsics,
            &intrinsics_inverse,
        )
        .unwrap();

        assert!((pose.translation.vector - expected_pose.translation.vector).norm() < 1e-2);
        assert!(pose.rotation.angle_to(&expected_pose.rotation) < 1e-2);
    }
}
