import numpy as np
import warp as wp


@wp.kernel
def compute_forward_nearest_neighbors(
    descriptors_a: wp.array(dtype=wp.float32, ndim=2),
    descriptors_b: wp.array(dtype=wp.float32, ndim=2),
    match_indices: wp.array(dtype=wp.int32),
    maximum_similarities: wp.array(dtype=wp.float32),
):
    index_a = wp.tid()
    number_of_b = descriptors_b.shape[0]
    dimensions = descriptors_a.shape[1]

    best_index = wp.int32(-1)
    best_similarity = wp.float32(-1e30)

    for index_b in range(number_of_b):
        current_similarity = wp.float32(0.0)
        for dimension_index in range(dimensions):
            current_similarity += (
                descriptors_a[index_a, dimension_index]
                * descriptors_b[index_b, dimension_index]
            )

        if current_similarity > best_similarity:
            best_similarity = current_similarity
            best_index = index_b

    match_indices[index_a] = best_index
    maximum_similarities[index_a] = best_similarity


@wp.kernel
def compute_backward_nearest_neighbors(
    descriptors_a: wp.array(dtype=wp.float32, ndim=2),
    descriptors_b: wp.array(dtype=wp.float32, ndim=2),
    match_indices: wp.array(dtype=wp.int32),
):
    index_b = wp.tid()
    number_of_a = descriptors_a.shape[0]
    dimensions = descriptors_b.shape[1]

    best_index = wp.int32(-1)
    best_similarity = wp.float32(-1e30)

    for index_a in range(number_of_a):
        current_similarity = wp.float32(0.0)
        for dimension_index in range(dimensions):
            current_similarity += (
                descriptors_b[index_b, dimension_index]
                * descriptors_a[index_a, dimension_index]
            )

        if current_similarity > best_similarity:
            best_similarity = current_similarity
            best_index = index_a

    match_indices[index_b] = best_index


@wp.kernel
def find_mutual_matches(
    forward_indices: wp.array(dtype=wp.int32),
    backward_indices: wp.array(dtype=wp.int32),
    maximum_similarities: wp.array(dtype=wp.float32),
    minimum_cosine_similarity: wp.float32,
    valid_mask: wp.array(dtype=wp.int32),
):
    index_a = wp.tid()
    index_b = forward_indices[index_a]

    is_valid = wp.int32(0)

    if index_b >= 0:
        if backward_indices[index_b] == index_a:
            if maximum_similarities[index_a] > minimum_cosine_similarity:
                is_valid = wp.int32(1)

    valid_mask[index_a] = is_valid


def match(
    descriptors_a: np.ndarray,
    descriptors_b: np.ndarray,
    minimum_cosine_similarity: float = 0.82,
) -> tuple[np.ndarray, np.ndarray]:

    # Data is transferred to the GPU to avoid O(N*M) memory allocations during dot products
    warp_descriptors_a = wp.array(descriptors_a, dtype=wp.float32)
    warp_descriptors_b = wp.array(descriptors_b, dtype=wp.float32)

    number_of_a = descriptors_a.shape[0]
    number_of_b = descriptors_b.shape[0]

    forward_indices = wp.empty(number_of_a, dtype=wp.int32)
    forward_similarities = wp.empty(number_of_a, dtype=wp.float32)
    backward_indices = wp.empty(number_of_b, dtype=wp.int32)
    valid_mask = wp.empty(number_of_a, dtype=wp.int32)

    wp.launch(
        kernel=compute_forward_nearest_neighbors,
        dim=number_of_a,
        inputs=[
            warp_descriptors_a,
            warp_descriptors_b,
            forward_indices,
            forward_similarities,
        ],
    )

    wp.launch(
        kernel=compute_backward_nearest_neighbors,
        dim=number_of_b,
        inputs=[warp_descriptors_a, warp_descriptors_b, backward_indices],
    )

    wp.launch(
        kernel=find_mutual_matches,
        dim=number_of_a,
        inputs=[
            forward_indices,
            backward_indices,
            forward_similarities,
            minimum_cosine_similarity,
            valid_mask,
        ],
    )

    # Arrays are moved back to the CPU because Warp currently lacks native dynamic array compression for boolean indexing
    numpy_forward_indices = forward_indices.numpy()
    numpy_valid_mask = valid_mask.numpy().astype(bool)

    index_0 = np.arange(number_of_a)[numpy_valid_mask]
    index_1 = numpy_forward_indices[numpy_valid_mask]

    return index_0, index_1
