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
        // Todo
    }
}
