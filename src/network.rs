use crate::micrograd;

pub struct LinearLayer {
    input_dim: usize,
    output_dim: usize,
    weights: Vec<micrograd::Value>,
}

impl LinearLayer {
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        let mut weights = Vec::new();
        for _row in 0..input_dim {
            for _col in 0..output_dim {
                weights.push(micrograd::Value::new(1.0))
            }
        }

        return Self {
            input_dim,
            output_dim,
            weights,
        };
    }

    pub fn forward(&self, input: Vec<micrograd::Value>) -> Vec<micrograd::Value> {
        assert_eq!(input.len(), self.input_dim);

        // Idea: for input [0, 1] (1x2) and weight matrix [0, 1, 2, 3, 4, 5] (2x3)
        // Create output: [0 + 1, 0 + 3, 0 + 5] (1x3)
        println!("{:?}", self.weights);
        let mut output_vec = Vec::new();
        for row in 0..self.output_dim {
            let mut sum = micrograd::Value::new(0.0);
            for col in 0..self.input_dim {
                let idx = self.input_dim * row + col;
                let to_add = &input[col] * &self.weights[idx];
                sum = &sum + &to_add;
            }
            output_vec.push(sum)
        }

        return output_vec;
    }
}
