use hnsw_rs::prelude::*;

pub struct HnswSlamMap<'a> {
    // The HNSW graph using standard f32 vectors and the Dot Product distance metric
    hnsw: Hnsw<'a, f32, DistDot>,
    next_landmark_id: usize,
}

impl HnswSlamMap<'_> {
    pub fn new(starting_id: usize) -> Self {
        // Standard parameters for a highly accurate HNSW graph
        let max_nb_connection = 16; // M parameter: higher = better recall, slower build
        let max_elements = 100_000; // Maximum capacity of the map
        let max_layer = 16; // Max layers in the graph
        let ef_construction = 200; // Search depth during graph construction

        let hnsw = Hnsw::<f32, DistDot>::new(
            max_nb_connection,
            max_elements,
            max_layer,
            ef_construction,
            DistDot {},
        );

        Self {
            hnsw,
            next_landmark_id: starting_id,
        }
    }

    /// Takes a normalized descriptor vector.
    /// Returns the existing landmark ID if a match is found.
    /// Otherwise, adds it to the map, assigns a new ID, and returns the new ID.
    pub fn get_or_add_landmark(&mut self, desc: Vec<f32>, match_threshold: f32) -> usize {
        // In hnsw_rs, DistDot converts Cosine Similarity to a distance metric
        // using roughly (1.0 - dot_product).
        // So a similarity threshold of 0.95 means the distance must be <= 0.05.
        let distance_threshold = 1.0 - match_threshold;

        // ef_search controls the recall/speed tradeoff during querying.
        let ef_search = 50;

        // 1. Search the graph for the single nearest neighbor
        // Note: hnsw_rs gracefully handles searches on an empty graph by returning an empty Vec.
        let neighbors = self.hnsw.search(&desc, 1, ef_search);

        if let Some(closest) = neighbors.first() {
            println!(
                "Closest neighbor found with distance {:.4} (threshold {:.4})",
                closest.distance, distance_threshold
            );
            if closest.distance <= distance_threshold {
                // MATCH FOUND: Return the existing ID!
                return closest.d_id;
            }
        }

        // 2. NO MATCH FOUND: We must add it as a new landmark.
        let new_id = self.next_landmark_id;
        self.next_landmark_id += 1;

        // hnsw_rs insert takes a tuple of (&[f32], DataId)
        self.hnsw.insert((&desc, new_id));

        new_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper function to ensure all test vectors are on the unit hypersphere
    fn normalize(mut v: Vec<f32>) -> Vec<f32> {
        let norm_sq: f32 = v.iter().map(|&x| x * x).sum();
        let norm = norm_sq.sqrt();
        if norm > 1e-8 {
            for val in v.iter_mut() {
                *val /= norm;
            }
        }
        v
    }

    #[test]
    fn test_standard_pipeline_hnsw() {
        // start ID counter at 0
        let mut map = HnswSlamMap::new(0);

        // match_threshold = 0.95 (must be 95% similar to be considered the SAME landmark)
        let match_threshold = 0.95;

        // 1. Add Vector 1 (Novel)
        let v1 = normalize(vec![1.0, 0.0, 0.0]);
        let id1 = map.get_or_add_landmark(v1.clone(), match_threshold);
        assert_eq!(id1, 0, "First novel vector should get starting ID 0");

        // 2. Add Vector 2 (A slightly noisy observation of v1)
        let v2 = normalize(vec![1.0, 0.05, 0.0]);
        let id2 = map.get_or_add_landmark(v2.clone(), match_threshold);
        assert_eq!(
            id2, 0,
            "Highly similar vector should return the existing ID 0"
        );

        // 3. Add Vector 3 (Completely novel)
        let v3 = normalize(vec![0.0, 1.0, 0.0]);
        let id3 = map.get_or_add_landmark(v3.clone(), match_threshold);
        assert_eq!(id3, 1, "Second novel vector should get new ID 2");

        // 4. Add Vector 4 (Completely novel)
        let v4 = normalize(vec![0.0, 0.0, 1.0]);
        let id4 = map.get_or_add_landmark(v4.clone(), match_threshold);
        assert_eq!(id4, 2, "Third novel vector should get new ID 2");

        // 5. Add Vector 5 (A slightly noisy observation of v3)
        let v5 = normalize(vec![0.0, 1.0, 0.05]);
        let id5 = map.get_or_add_landmark(v5.clone(), match_threshold);
        assert_eq!(
            id5, 1,
            "Highly similar vector to v3 should return existing ID 1"
        );
    }

    #[test]
    fn test_many_landmarks_hnsw() {
        // start ID counter at 0
        let mut map = HnswSlamMap::new(0);
        let num_iterations = 5000;

        let mut assigned_ids = Vec::with_capacity(num_iterations);

        // Generate 5000 pseudo-random normalized descriptors
        for i in 0..num_iterations {
            let x = (i as f32 * 0.1).sin();
            let y = (i as f32 * 0.1).cos();
            let z = ((i % 10) as f32 * 0.1).sin();
            let v = normalize(vec![x, y, z]);

            let id = map.get_or_add_landmark(v, 0.99);
            assigned_ids.push(id);
        }

        // Pick an arbitrary frame we processed (e.g., the 2500th descriptor)
        let target_idx = 2500;
        let expected_id = assigned_ids[target_idx];

        // Recreate the exact descriptor from that frame
        let x = (target_idx as f32 * 0.1).sin();
        let y = (target_idx as f32 * 0.1).cos();
        let z = ((target_idx % 10) as f32 * 0.1).sin();
        let query_vec = normalize(vec![x, y, z]);

        // Pass it through the standard pipeline again
        let retrieved_id = map.get_or_add_landmark(query_vec, 0.99);

        // The HNSW graph must traverse the layers and successfully return the exact same ID
        assert_eq!(retrieved_id, expected_id);
    }
}
