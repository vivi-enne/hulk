use factrs::{
    containers::{Key, ValuesOrder},
    core::{GaussNewton, Values},
    traits::Optimizer,
    variables::{SE23, VariableSafe},
};
use faer::{
    Mat, Par,
    linalg::solvers::Solve,
    sparse::{SparseColMat, linalg::matmul::sparse_sparse_matmul},
};
// use foldhash::HashMap;
use std::{
    collections::{BTreeSet, HashMap, HashSet},
    ops::Mul,
};

pub fn marginalize(
    optimizer: &mut GaussNewton,
    values: &mut Values,
    smallest_interval_index_in_window: u32,
) {
    let keys_to_marginalize = find_keys_to_marginalize(values, smallest_interval_index_in_window);
    let (value_order, keep_offset) = resort_value_order(values, &keys_to_marginalize);
    let graph_order = optimizer.graph().sparsity_pattern(value_order);
    let linearized_graph = optimizer.graph().linearize(values);

    let residual_jacobian = linearized_graph.residual_jacobian(&graph_order);
    let linear_system = from_jacobian(residual_jacobian.diff, residual_jacobian.value);
    let h_dense = linear_system.h.to_dense();
    let b = linear_system.b;

    let (h_mm, h_mk, h_km, h_kk) = h_dense.split_at(keep_offset, keep_offset);
    let (b_m, b_k) = b.split_at_row(keep_offset);

    let lu = h_mm.partial_piv_lu();

    let solved_h = lu.solve(&h_mk);
    let solved_b = lu.solve(&b_m);

    let h_prior = &h_kk - &h_km * solved_h;
    let b_prior = &b_k - &h_km * solved_b;

    // let prior = MarginalPriorFactor {
    //     keys: keys_to_keep.clone(),
    //     linearization_point: extract_linearization_point(values, &keys_to_keep),
    //     a,
    //     b: d,
    // };

    //todo remove factors touching marginalized values (start time is smaller than smallest_interval_index_in_window)
    let keys_to_marginalize_set: HashSet<_> =
        keys_to_marginalize.into_iter().map(|(k, _)| *k).collect();

    let removed_factors = optimizer.graph_mut().remove_factors(|factor| {
        factor
            .keys()
            .iter()
            .any(|key| keys_to_marginalize_set.contains(key))
    });

    // check if staete touches removed factor and is not marginalized
    let boundary_keys = removed_factors
        .iter()
        .flat_map(|factor| {
            factor
                .keys()
                .iter()
                .filter(|key| !keys_to_marginalize_set.contains(key))
        })
        .copied()
        .collect::<HashSet<_>>();

    values.retain(|value| !keys_to_marginalize_set.contains(value));

    // todo: factor
}

fn find_keys_to_marginalize(
    values: &Values,
    smallest_interval_index_in_window: u32,
) -> Vec<(&Key, &Box<dyn VariableSafe + 'static>)> {
    values
        .iter()
        .filter(|(key, value)| {
            value.is::<SE23>() && key.0 < smallest_interval_index_in_window.into()
        })
        .collect()
}

fn resort_value_order(
    values: &Values,
    keys_to_marginalize: &[(&Key, &Box<dyn VariableSafe + 'static>)],
) -> (ValuesOrder, usize) {
    let mut map = HashMap::default();
    let mut offset = 0usize;

    for (key, value) in keys_to_marginalize {
        let dim = value.dim();
        map.insert(**key, factrs::containers::Idx { idx: offset, dim });
        offset += dim;
    }
    let keep_offset = offset;

    for (key, value) in values.iter() {
        if map.contains_key(key) {
            continue;
        }
        let dim = value.dim();
        map.insert(*key, factrs::containers::Idx { idx: offset, dim });
        offset += dim;
    }

    (ValuesOrder::new(map), keep_offset)
}
pub struct LinearSystem {
    h: SparseColMat<usize, f64>,
    b: Mat<f64>,
}

pub fn from_jacobian(j: SparseColMat<usize, f64>, r: Mat<f64>) -> LinearSystem {
    // TODO: is this correct?
    let jt = j.transpose().to_col_major().unwrap();
    let h = sparse_sparse_matmul(jt.as_ref(), j.as_ref(), 1.0, Par::Seq).unwrap();
    let b = jt.mul(r);

    LinearSystem { h, b }
}

pub struct MarginalPriorFactor {
    pub keys: Vec<Key>,
    pub linearization_point: Vec<f64>,
    pub a: Mat<f64>,
    pub b: Mat<f64>,
}

fn variable_dim(values: &Values, key: Key) -> usize {
    todo!("for later when also using other variable types")
}
