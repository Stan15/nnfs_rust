use ndarray::prelude::*;

fn main() {
    let inputs = array![1.0, 2.0, 3.0, 2.5];
    let weights = array![
        [0.2, 0.8, -0.5, 1.0],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87]
    ];
    let biases = array![2.0, 3.0, 0.5];

    let output = weights.dot(&inputs) + biases;
    println!("{}", output);
}
