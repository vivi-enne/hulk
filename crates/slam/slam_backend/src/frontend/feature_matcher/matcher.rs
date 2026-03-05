use ordered_float::OrderedFloat;

use crate::frontend::feature_matcher::matching::CosineSimilarity;

#[derive(Debug)]
pub struct CosineSimilarityMatcher {
    threshold: f32,
}

impl CosineSimilarityMatcher {
    pub fn new(threshold: f32) -> Self {
        Self { threshold }
    }

    pub fn mutual_nearest_neighbor_match<Descriptor: CosineSimilarity>(
        &self,
        source: &[Descriptor],
        target: &[Descriptor],
    ) -> (Vec<usize>, Vec<usize>) {
        let distances = DistanceTable::build(source, target);
        let source_to_target: Vec<usize> = (0..source.len())
            .filter_map(|i| distances.max_index_of_target(i, self.threshold))
            .collect();
        let target_to_source: Vec<usize> = (0..target.len())
            .filter_map(|i| distances.max_index_of_source(i, self.threshold))
            .collect();

        let mut source_matches = Vec::with_capacity(target_to_source.len());
        let mut target_matches = Vec::with_capacity(target_to_source.len());

        for (source_index, target_index) in source_to_target.into_iter().enumerate() {
            if target_to_source[target_index] == source_index {
                source_matches.push(source_index);
                target_matches.push(target_index);
            }
        }

        (source_matches, target_matches)
    }
}

#[derive(Debug)]
struct DistanceTable {
    source_length: usize,
    target_length: usize,
    distances: Vec<f32>,
}

impl DistanceTable {
    pub fn build<Descriptor: CosineSimilarity>(
        source: &[Descriptor],
        target: &[Descriptor],
    ) -> Self {
        let mut distances = Vec::with_capacity(source.len() * target.len());

        for a in source {
            for b in target {
                distances.push(a.cosine_similarity(b));
            }
        }

        Self {
            source_length: source.len(),
            target_length: target.len(),
            distances,
        }
    }

    pub fn distance(&self, source_idx: usize, target_idx: usize) -> f32 {
        let index = source_idx * self.target_length + target_idx;
        self.distances[index]
    }

    pub fn max_index_of_target(&self, source_idx: usize, threshold: f32) -> Option<usize> {
        (0..self.target_length).max_by_key(|j| {
            let value = self.distance(source_idx, *j);
            if value < threshold {
                return None;
            }
            Some(OrderedFloat(value))
        })
    }

    pub fn max_index_of_source(&self, target_idx: usize, threshold: f32) -> Option<usize> {
        (0..self.source_length).max_by_key(|i| {
            let value = self.distance(*i, target_idx);
            if value < threshold {
                return None;
            }
            Some(OrderedFloat(value))
        })
    }
}
