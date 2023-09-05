use ndarray::Array2;

struct ActivationFunctions {
    pub relu: fn(Array2<f64>) -> Array2<f64>,
    pub relu_backward: fn(Array2<f64>, Array2<f64>) -> Array2<f64>,
    pub logsoftmax: fn(Array2<f64>) -> Array2<f64>,
    pub logsoftmax_backward: fn(Array2<f64>, Array2<f64>) -> Array2<f64>,
}

impl ActivationFunctions {
    pub fn new() -> Self {
        Self {
            relu: Self::relu,
            relu_backward: Self::relu_backward,
            logsoftmax: Self::logsoftmax,
            logsoftmax_backward: Self::logsoftmax_backward,
        }
    }
    pub fn relu(x: Array2<f64>) -> Array2<f64> {
        x.mapv(|x| if x > 0.0 { x } else { 0.0 })
    }

    pub fn relu_backward(x: Array2<f64>, y: Array2<f64>) -> Array2<f64> {
        x.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 }) * y
    }

    pub fn logsoftmax(x: Array2<f64>) -> Array2<f64> {
        let max_x = x.fold_axis(ndarray::Axis(1), f64::NAN, |&a, &b| a.max(b));
        let max_x = max_x.insert_axis(ndarray::Axis(1));
        &x - &max_x
            - &((x - &max_x)
                .mapv(f64::exp)
                .sum_axis(ndarray::Axis(1))
                .mapv(f64::ln)
                .insert_axis(ndarray::Axis(1)))
    }

    pub fn logsoftmax_backward(x: Array2<f64>, y: Array2<f64>) -> Array2<f64> {
        let softmax_x = (&x
            - &x.fold_axis(ndarray::Axis(1), f64::NAN, |&a, &b| a.max(b))
                .insert_axis(ndarray::Axis(1)))
            .mapv(f64::exp);
        let softmax_sum = softmax_x
            .sum_axis(ndarray::Axis(1))
            .insert_axis(ndarray::Axis(1));
        let softmax = softmax_x / &softmax_sum;
        let n = x.shape()[1];
        let delta_ij = Array2::eye(n);
        let softmax_matrix = softmax
            .insert_axis(ndarray::Axis(1))
            .broadcast((n, n))
            .unwrap()
            .to_owned();
        let derivative = &delta_ij - &softmax_matrix;
        y.dot(&derivative)
    }
}

// pub struct Model {
//     pub weights: Array2<Array2<f32>>,
//     pub biases: Array2<f32>,
// }

// impl Model {
//     pub fn new(weights: Array2<Array2<f32>>, biases: Array2<f32>) -> Self {
//         Self { weights, biases }
//     }

//     pub fn forward(&self, input: Array2<f32>) -> Array2<f32> {
//         let mut result = input.clone();
//         for i in 0..self.weights.shape()[0] {
//             result = self.weights[[i, 0]].dot(&result) + &self.biases[[i, 0]];
//             result = ActivationFunctions::sigmoid(result);
//         }
//         result
//     }

//     fn cost(&self, input: Array2<f32>, output: Array2<f32>) -> f32 {
//         let mut result = 0.0;
//         let mut prediction = self.forward(input);
//         prediction = ActivationFunctions::softmax(prediction);
//         for i in 0..output.shape()[0] {
//             result += output[[i, 0]] * prediction[[i, 0]].ln();
//         }
//         -result
//     }

//     pub fn backward(
//         &self,
//         input: Array2<f32>,
//         output: Array2<f32>,
//     ) -> (Array2<Array2<f32>>, Array2<f32>) {
//         let mut weights_gradient = Array2::zeros(self.weights.shape());
//         let mut biases_gradient = Array2::zeros(self.biases.shape());
//         let mut prediction = self.forward(input);
//         prediction = ActivationFunctions::softmax(prediction);
//         let mut delta = prediction - output;
//         for i in (0..self.weights.shape()[0]).rev() {
//             weights_gradient[[i, 0]] = delta.dot(&input.t());
//             biases_gradient[[i, 0]] = delta.sum();
//             delta = self.weights[[i, 0]].t().dot(&delta)
//                 * ActivationFunctions::sigmoid_backward(prediction);
//         }
//         (weights_gradient, biases_gradient)
//     }
// }

pub struct Model {
    pub weights: Array2<Array2<f32>>,
    pub biases: Array2<f32>,
}

impl Model {
    pub fn new(weights: Array2<Array2<f32>>, biases: Array2<f32>) -> Self {
        Self { weights, biases }
    }
}
