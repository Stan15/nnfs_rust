use std::iter::zip;

const INPUT_SIZE:usize = 4;
const OUTPUT_SIZE:usize = 3;
fn main() {
    let inputs: [f64; INPUT_SIZE] = [1.0, 2.0, 3.0, 2.5];

    let weights_set: [[f64;INPUT_SIZE];OUTPUT_SIZE] = [
        [0.2, 0.8, -0.5, 1.0],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87]
    ];
    let biases = [2.0, 3.0, 0.5];

    let mut layers_output: [f64; OUTPUT_SIZE] = [0.0; OUTPUT_SIZE];
    for (neuron, (weights, bias)) in zip(weights_set, biases).enumerate() {
        for (input, weight) in zip(inputs, weights) {
            layers_output[neuron] += input*weight;
        }
        layers_output[neuron] += bias;
    }

    println!("{:?}", layers_output);
}
