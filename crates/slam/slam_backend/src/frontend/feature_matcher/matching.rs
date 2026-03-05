use nalgebra::{Dim, Storage, U1, Unit, Vector};

pub trait CosineSimilarity {
    fn cosine_similarity(&self, other: &Self) -> f32;
}

impl<D, S> CosineSimilarity for Unit<Vector<f32, D, S>>
where
    D: Dim,
    S: Storage<f32, D, U1>,
{
    fn cosine_similarity(&self, other: &Self) -> f32 {
        self.dot(other)
    }
}
