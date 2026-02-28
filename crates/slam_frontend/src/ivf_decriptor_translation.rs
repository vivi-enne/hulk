use std::collections::HashMap;

/// Computes the dot product of two L2-normalized vectors (Cosine Similarity).
#[inline]
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub struct DynamicSlamMap {
    centroids: Vec<Vec<f32>>,
    buckets: HashMap<usize, Vec<(usize, Vec<f32>)>>,
    cluster_threshold: f32, // Minimum similarity to join a centroid's bucket (e.g., 0.75)
    next_landmark_id: usize, // Internal counter for assigning new IDs
}

impl DynamicSlamMap {
    pub fn new(cluster_threshold: f32, starting_id: usize) -> Self {
        Self {
            centroids: Vec::new(),
            buckets: HashMap::new(),
            cluster_threshold,
            next_landmark_id: starting_id,
        }
    }

    /// Takes a normalized descriptor vector.
    /// Returns the existing landmark ID if a match is found.
    /// Otherwise, adds it to the map, assigns a new ID, and returns the new ID.
    pub fn get_or_add_landmark(&mut self, desc: Vec<f32>, match_threshold: f32) -> usize {
        // Handle the very first descriptor inserted into the map
        if self.centroids.is_empty() {
            return self.add_as_new_centroid(desc);
        }

        // 1. Find the closest cluster center (Centroid)
        let mut best_centroid_idx = 0;
        let mut best_centroid_sim = -1.0;

        for (idx, centroid) in self.centroids.iter().enumerate() {
            let similarity = dot_product(&desc, centroid);
            if similarity > best_centroid_sim {
                best_centroid_sim = similarity;
                best_centroid_idx = idx;
            }
        }

        // 2. Search inside that specific cluster's bucket for an exact landmark match
        if let Some(bucket) = self.buckets.get(&best_centroid_idx) {
            let mut best_match_id = None;
            let mut best_match_sim = -1.0;

            for (id, existing_desc) in bucket {
                let sim = dot_product(&desc, existing_desc);
                // If it beats the threshold AND is the best match so far
                if sim >= match_threshold && sim > best_match_sim {
                    best_match_sim = sim;
                    best_match_id = Some(*id);
                }
            }

            // If we found a match that satisfies the threshold, return its ID!
            if let Some(matched_id) = best_match_id {
                return matched_id;
            }
        }

        // 3. NO MATCH FOUND: We must add it as a new landmark.
        let new_id = self.next_landmark_id;
        self.next_landmark_id += 1;

        // Decide where this new landmark lives: in the best existing bucket, or a new one?
        if best_centroid_sim >= self.cluster_threshold {
            // It belongs in the closest existing bucket
            self.buckets.entry(best_centroid_idx).or_default().push((new_id, desc));
        } else {
            // It is entirely novel (e.g., camera turned a corner). Create a new centroid.
            let new_centroid_idx = self.centroids.len();
            self.centroids.push(desc.clone());
            self.buckets.insert(new_centroid_idx, vec![(new_id, desc)]);
        }

        new_id
    }

    /// Deletes a landmark when your SLAM backend optimizer culls it.
    pub fn cull_landmark(&mut self, target_id: usize, desc_hint: &[f32]) {
        // Find which bucket it likely lives in using the descriptor hint
        let mut best_centroid_idx = 0;
        let mut best_centroid_sim = -1.0;

        for (idx, centroid) in self.centroids.iter().enumerate() {
            let sim = dot_product(desc_hint, centroid);
            if sim > best_centroid_sim {
                best_centroid_sim = sim;
                best_centroid_idx = idx;
            }
        }

        // Remove the target ID from that bucket
        if let Some(bucket) = self.buckets.get_mut(&best_centroid_idx) {
            bucket.retain(|(id, _)| *id != target_id);
        }
    }

    // Helper method for the first insertion
    fn add_as_new_centroid(&mut self, desc: Vec<f32>) -> usize {
        let new_id = self.next_landmark_id;
        self.next_landmark_id += 1;
        
        self.centroids.push(desc.clone());
        self.buckets.insert(0, vec![(new_id, desc)]);
        
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
    fn test_standard_pipeline() {
        // cluster_threshold = 0.8 (when to join a cluster's search bucket)
        // starting_id = 1 (our first landmark will be ID 0)
        let mut map = DynamicSlamMap::new(0.8, 0);
        
        // match_threshold = 0.95 (must be 95% similar to be considered the SAME landmark)
        let match_threshold = 0.99; 

        // 1. Add Vector 1 (Novel)
        let v1 = normalize(vec![1.0, 0.0, 0.0]);
        let id1 = map.get_or_add_landmark(v1.clone(), match_threshold);
        assert_eq!(id1, 0, "First novel vector should get starting ID 0");

        // 2. Add Vector 2 (A slightly noisy observation of v1)
        // Similarity to v1 will be > 0.95, so it should match, not add.
        let v2 = normalize(vec![1.0, 0.05, 0.0]);
        let id2 = map.get_or_add_landmark(v2.clone(), match_threshold);
        assert_eq!(id2, 0, "Highly similar vector should return the existing ID 0");

        // 3. Add Vector 3 (Completely novel, orthogonal)
        let v3 = normalize(vec![0.0, 1.0, 0.0]);
        let id3 = map.get_or_add_landmark(v3.clone(), match_threshold);
        assert_eq!(id3, 1, "Second novel vector should get new ID 1");

        // 4. Add Vector 4 (Completely novel, orthogonal to both)
        let v4 = normalize(vec![0.0, 0.0, 1.0]);
        let id4 = map.get_or_add_landmark(v4.clone(), match_threshold);
        assert_eq!(id4, 2, "Third novel vector should get new ID 2");

        // 5. Add Vector 5 (A slightly noisy observation of v3)
        let v5 = normalize(vec![0.0, 1.0, 0.1]);
        let id5 = map.get_or_add_landmark(v5.clone(), match_threshold);
        assert_eq!(id5, 1, "Highly similar vector to v3 should return existing ID 1");

        // Verify internal state: Despite 5 pipeline calls, only 3 unique landmarks exist
        let total_stored_landmarks: usize = map.buckets.values().map(|b| b.len()).sum();
        assert_eq!(total_stored_landmarks, 3);
    }

    #[test]
    fn test_many_landmarks() {
        // cluster_threshold = 0.75, starting_id = 0
        let mut map = DynamicSlamMap::new(0.75, 0);
        let num_iterations = 5000;
        
        // We will store the IDs assigned during the initial run to verify them later
        let mut assigned_ids = Vec::with_capacity(num_iterations);
        
        // Generate 5000 pseudo-random normalized descriptors
        for i in 0..num_iterations {
            let x = (i as f32 * 0.1).sin();
            let y = (i as f32 * 0.1).cos();
            let z = ((i % 10) as f32 * 0.1).sin();
            let v = normalize(vec![x, y, z]);
            
            // Using a strict match threshold ensures most vectors get their own ID,
            // though some periodic overlaps will naturally match and share IDs.
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
        
        // The map must successfully route through the centroids and buckets
        // to return the exact same landmark ID it assigned previously.
        assert_eq!(retrieved_id, expected_id);
    }

    #[test]
    fn test_landmark_culling() {
        let mut map = DynamicSlamMap::new(0.8, 100);
        let v1 = normalize(vec![1.0, 1.0, 1.0]);
        
        // Add it and get the ID (should be 100)
        let id1 = map.get_or_add_landmark(v1.clone(), 0.95);
        assert_eq!(id1, 100);
        
        // Simulate the SLAM backend deciding this landmark is bad
        map.cull_landmark(100, &v1);
        
        // If we query the same vector again, it shouldn't match ID 100 anymore.
        // It should act like a novel vector and get the next available ID (101).
        let id2 = map.get_or_add_landmark(v1.clone(), 0.95);
        assert_eq!(id2, 101);
    }
}