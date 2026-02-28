// input: descriptor vector with 65 outputs
// output: feature id (int)

// use balltree to match those, features are matching if they are in a given radius to each other
// if no matching feature in the ball tree, create a new one with a new id and add it to the tree

use std::f32::INFINITY;

use ball_tree::BallTree;

fn translate_descriptor_to_feature_id(descriptor: Vec<f32>) -> u32 {
    let points = vec![];
    let values = vec![];
    let tree = BallTree::new(points, values.clone());

    let mut query = tree.query();

    let max_radius = 0.1; // example radius, adjust as needed
    let smallest_distance = INFINITY;
    let best_value = None;

    // returns (p,d,v) iterator of points, distances and values within the radius
    query
        .nn_within(&descriptor, max_radius)
        .for_each(|(point, distance, value)| {
            if distance < smallest_distance {
                smallest_distance = distance;
                best_value = Some(value);
            }
        });

    
    match best_value {
        Some(value) => value, 
        None => {
            // No matching feature found, create a new one
            let new_id = values.len() as u32; // example id generation, adjust as needed
            tree.insert(descriptor, new_id);
            new_id
        }
    }
}
