use ndarray::Array1;
use ndarray::Array2;

struct ActivationFunctions {}

impl ActivationFunctions {
    pub fn relu1d(x: Array1<f64>) -> Array1<f64> {
        x.mapv(|x| if x > 0.0 { x } else { 0.0 })
    }

    pub fn relu2d(x: Array2<f64>) -> Array2<f64> {
        x.mapv(|x| if x > 0.0 { x } else { 0.0 })
    }

    pub fn relu_backward1d(x: Array1<f64>, y: Array1<f64>) -> Array1<f64> {
        x.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 }) * y
    }

    pub fn relu_backward2d(x: Array2<f64>, y: Array2<f64>) -> Array2<f64> {
        x.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 }) * y
    }

    pub fn logsoftmax1d(x: Array1<f64>) -> Array1<f64> {
        let max_x = x.fold(f64::NAN, |a, b| a.max(*b));
        let max_x = Array1::from_elem(x.len(), max_x);
        &x - &max_x - Array1::from_elem(x.len(), x.mapv(f64::exp).sum().ln())
    }

    pub fn logsoftmax2d(x: Array2<f64>) -> Array2<f64> {
        let max_x = x.fold_axis(ndarray::Axis(1), f64::NAN, |&a, &b| a.max(b));
        let max_x = max_x.insert_axis(ndarray::Axis(1));
        &x - &max_x
            - &((x - &max_x)
                .mapv(f64::exp)
                .sum_axis(ndarray::Axis(1))
                .mapv(f64::ln)
                .insert_axis(ndarray::Axis(1)))
    }

    pub fn logsoftmax_backward1d(x: Array1<f64>, y: Array1<f64>) -> Array1<f64> {
        let softmax_x = (&x - x.fold(f64::NAN, |a, b| a.max(*b))).mapv(f64::exp);
        let softmax_sum = softmax_x.sum();
        let softmax = softmax_x / softmax_sum;
        let n = x.len();
        let delta_ij = Array2::eye(n);
        let softmax_matrix = softmax.broadcast(n).unwrap().to_owned();
        let derivative = &delta_ij - &softmax_matrix;
        y.dot(&derivative)
    }

    pub fn logsoftmax_backward2d(x: Array2<f64>, y: Array2<f64>) -> Array2<f64> {
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

pub struct Model {
    pub weights: (Array2<f64>, Array2<f64>),
    pub learning_rates: (f64, f64),
}

impl Model {
    pub fn new(weights: (Array2<f64>, Array2<f64>), learning_rates: (f64, f64)) -> Self {
        Self {
            weights,
            learning_rates,
        }
    }

    pub fn infer(&self, input: Vec<f64>) -> i32 {
        let input = Array1::from(input);
        let mut layer = self.weights.0.dot(&input);
        layer = ActivationFunctions::relu1d(layer);
        layer = self.weights.1.dot(&layer);
        layer
            .iter()
            .enumerate()
            .fold(
                (0, f64::NAN),
                |(i, max), (j, &x)| {
                    if x > max {
                        (j, x)
                    } else {
                        (i, max)
                    }
                },
            )
            .0 as i32
    }

    pub fn train1d(&mut self, input: Vec<f64>, target: i32) -> f64 {
        let input = Array1::from(input);
        let mut layer = self.weights.0.dot(&input);
        layer = ActivationFunctions::relu1d(layer);
        layer = self.weights.1.dot(&layer);
        layer = ActivationFunctions::logsoftmax1d(layer);
        let mut target_vec = vec![0.0; 10];
        target_vec[target as usize] = 1.0;
        let target = Array1::from(target_vec);
        let loss = -(&target * &layer).sum();
        let mut gradients = ActivationFunctions::logsoftmax_backward1d(layer, target);
        self.weights.1 = &self.weights.1 - &gradients * self.learning_rates.1;
        gradients = ActivationFunctions::relu_backward1d(self.weights.1.dot(&input), gradients);
        self.weights.0 = &self.weights.0 - &gradients * self.learning_rates.0;
        loss
    }

    pub fn train2d(&mut self, input: Vec<Vec<f64>>, target: Vec<i32>) -> f64 {
        let input = Array2::from_shape_vec(
            (input.len(), input[0].len()),
            input.into_iter().flatten().collect(),
        )
        .unwrap();
        let mut layer = self.weights.0.dot(&input);
        layer = ActivationFunctions::relu2d(layer);
        layer = self.weights.1.dot(&layer);
        layer = ActivationFunctions::logsoftmax2d(layer);
        let mut target_vec = vec![vec![0.0; 10]; target.len()];
        for (i, t) in target.iter().enumerate() {
            target_vec[i][*t as usize] = 1.0;
        }
        let target = Array2::from_shape_vec(
            (target_vec.len(), target_vec[0].len()),
            target_vec.into_iter().flatten().collect(),
        )
        .unwrap();
        let loss = -(&target * &layer).sum();
        let mut gradients = ActivationFunctions::logsoftmax_backward2d(layer, target);
        self.weights.1 = &self.weights.1 - &gradients * self.learning_rates.1;
        gradients = ActivationFunctions::relu_backward2d(self.weights.1.dot(&input), gradients);
        self.weights.0 = &self.weights.0 - &gradients * self.learning_rates.0;
        loss
    }
}
