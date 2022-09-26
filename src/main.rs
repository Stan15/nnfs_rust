use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

struct LayerDense {
    weights: Array2<f64>,
    biases: Array1<f64>,
}

impl LayerDense {
    fn new(n_inputs: i32, n_neurons: i32) -> Self {
        Self {
            weights: 0.10 * Array2::random((n_inputs, n_neurons), Uniform::new(0., 10.)),
            biases: Array2::zeros((1, n_neurons)),
        }
    }
    
    fn forward(&self, &inputs: &Array2<f64>) -> Array2<f64> {
        &inputs.dot(&self.weights) + &self.biases
    }
}

fn main() {
    let X = array![
        [1., 2., 3., 2.5],
        [2., 5., -1., 2.],
        [-1.5, 2.7, 3.3, -0.8]
    ];

    let layer1 = LayerDense::new(4, 5);
    let layer2 = LayerDense::new(5, 2);

    let layer1_output = layer1.forward(&X);
    let layer2_output = layer2.forward(&layer1_output);
    println!("{:8.4}", layer2_output);
}
