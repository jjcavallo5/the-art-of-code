use std;
use std::hash::*;

use crate::micrograd;

pub struct LinearLayer {
    input_dim: usize,
    output_dim: usize,
    weights: Vec<micrograd::Value>,
    biases: Vec<micrograd::Value>,
}

impl LinearLayer {
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        return Self {
            input_dim,
            output_dim,
            weights: init_weights(input_dim, output_dim),
            biases: init_weights(output_dim, 1),
        };
    }

    pub fn forward(&self, input: Vec<micrograd::Value>) -> Vec<micrograd::Value> {
        assert_eq!(input.len(), self.input_dim);

        // Idea: for input [0, 1] (1x2) and weight matrix [0, 1, 2, 3, 4, 5] (2x3)
        // Create output: [0 + 1, 0 + 3, 0 + 5] (1x3)
        let mut dot_prod = Vec::new();
        for row in 0..self.output_dim {
            let mut sum = micrograd::Value::new(0.0);
            for col in 0..self.input_dim {
                let idx = self.input_dim * row + col;
                let to_add = &input[col] * &self.weights[idx];
                sum = &sum + &to_add;
            }
            dot_prod.push(sum);
        }

        let output_vec = dot_prod
            .iter()
            .zip(&self.biases)
            .map(|tup| tup.0 + &tup.1)
            .collect();

        return output_vec;
    }
    pub fn parameters(&self) -> (&Vec<micrograd::Value>, &Vec<micrograd::Value>) {
        return (&self.weights, &self.biases);
    }
}

fn random() -> u64 {
    let hasher = std::hash::RandomState::new().build_hasher();
    return hasher.finish();
}

fn init_weights(input_s: usize, output_s: usize) -> Vec<micrograd::Value> {
    let mut rand_vec = Vec::new();
    for _ in 0..input_s * output_s {
        let value = random();
        rand_vec.push(value);
    }

    let max = *rand_vec.iter().max().expect("BOOM!") as f64;

    return rand_vec
        .iter()
        .map(|v| (*v as f64) / max)
        .map(|v| v - 0.5)
        .map(|v| v * (1.0 / input_s as f64).sqrt())
        .map(|v| micrograd::Value::new(v))
        .collect();
}
