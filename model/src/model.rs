use ndarray::Array2;

pub struct ActivationFunctions {
    pub sigmoid: fn(f32) -> f32,
    pub sigmoid_backward: fn(f32) -> f32,
    pub relu: fn(f32) -> f32,
    pub relu_backward: fn(f32) -> f32,
    pub softmax: fn(Array2<f32>) -> Array2<f32>,
    pub softmax_backward: fn(Array2<f32>) -> Array2<f32>,
}

pub impl ActivationFunctions {
    pub fn new() -> Self {
        Self {
            sigmoid,
            sigmoid_backward,
            relu,
            relu_backward,
            softmax,
            softmax_backward,
        }
    }
    pub fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    pub fn sigmoid_backward(x: f32) -> f32 {
        x * (1.0 - x)
    }

    pub fn relu(x: f32) -> f32 {
        if x > 0.0 {
            x
        } else {
            0.0
        }
    }

    pub fn relu_backward(x: f32) -> f32 {
        if x > 0.0 {
            1.0
        } else {
            0.0
        }
    }

    pub fn softmax(x: Array2<f32>) -> Array2<f32> {
        let mut sum = 0.0;
        let mut result = Array2::zeros(x.shape());
        for i in 0..x.shape()[0] {
            sum += x[[i, 0]].exp();
        }
        for i in 0..x.shape()[0] {
            result[[i, 0]] = x[[i, 0]].exp() / sum;
        }
        result
    }

    pub fn softmax_backward(x: Array2<f32>) -> Array2<f32> {
        let mut result = Array2::zeros(x.shape());
        for i in 0..x.shape()[0] {
            result[[i, 0]] = x[[i, 0]] * (1.0 - x[[i, 0]]);
        }
        result
    }
}

pub struct Model {
    pub weights: Array2<Array2<f32>>,
    pub biases: Array2<f32>,
    activation_functions: ActivationFunctions,
}

pub impl Model {
    pub fn new(weights: Array2<Array2<f32>>, biases: Array2<f32>) -> Self {
        Self {
            weights,
            biases,
            activation_functions: ActivationFunctions::new(),
        }
    }

    pub fn forward(&self, input: Array2<f32>) -> Array2<f32> {
        let mut result = input.clone();
        for i in 0..self.weights.shape()[0] {
            result = self.weights[[i, 0]].dot(&result) + &self.biases[[i, 0]];
            result = self.activation_functions.sigmoid(result);
        }
        result
    }

    fn cost(&self, input: Array2<f32>, output: Array2<f32>) -> f32 {
        let mut result = 0.0;
        let mut prediction = self.forward(input);
        prediction = self.activation_functions.softmax(prediction);
        for i in 0..output.shape()[0] {
            result += output[[i, 0]] * prediction[[i, 0]].ln();
        }
        -result
    }

    pub fn backward(
        &self,
        input: Array2<f32>,
        output: Array2<f32>,
    ) -> (Array2<Array2<f32>>, Array2<f32>) {
        let mut weights_gradient = Array2::zeros(self.weights.shape());
        let mut biases_gradient = Array2::zeros(self.biases.shape());
        let mut prediction = self.forward(input);
        prediction = self.activation_functions.softmax(prediction);
        let mut delta = prediction - output;
        for i in (0..self.weights.shape()[0]).rev() {
            weights_gradient[[i, 0]] = delta.dot(&input.t());
            biases_gradient[[i, 0]] = delta.sum();
            delta = self.weights[[i, 0]].t().dot(&delta)
                * self.activation_functions.sigmoid_backward(input);
        }
        (weights_gradient, biases_gradient)
    }
}
