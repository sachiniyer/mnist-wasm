use crate::activations::ActivationFunctions;
use ndarray::Array1;
use ndarray::Array2;

#[derive(Clone, Debug)]
pub struct Model {
    pub weights: (Array2<f64>, Array2<f64>),
    pub learning_rates: (f64, f64),
}

impl Model {
    pub fn new(weights: (Vec<Vec<f64>>, Vec<Vec<f64>>), learning_rates: (f64, f64)) -> Self {
        Self {
            weights: (
                Array2::from_shape_vec(
                    (weights.0.len(), weights.0[0].len()),
                    weights.0.into_iter().flatten().collect(),
                )
                .unwrap(),
                Array2::from_shape_vec(
                    (weights.1.len(), weights.1[0].len()),
                    weights.1.into_iter().flatten().collect(),
                )
                .unwrap(),
            ),
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
        let layer1 = self.weights.0.dot(&input);
        let layer1_relu = ActivationFunctions::relu1d(layer1);
        let layer2 = self.weights.1.dot(&layer1_relu);
        let output = ActivationFunctions::logsoftmax1d(layer2);
        let mut target_vec = vec![0.0; 10];
        target_vec[target as usize] = 1.0;
        let target = Array1::from(target_vec);
        let loss = -(&target * &output).sum();
        let layer2_gradients = ActivationFunctions::logsoftmax_backward1d(output, target);
        self.weights.1 = &self.weights.1 - &layer2_gradients * self.learning_rates.1;
        let layer1_gradients =
            ActivationFunctions::relu_backward1d(self.weights.1.dot(&input), layer2_gradients);
        self.weights.0 = &self.weights.0 - &layer1_gradients * self.learning_rates.0;
        loss
    }

    pub fn train2d(&mut self, input: Vec<Vec<f64>>, target: Vec<i32>) -> f64 {
        let input = Array2::from_shape_vec(
            (input.len(), input[0].len()),
            input.into_iter().flatten().collect(),
        )
        .unwrap();
        let layer1 = self.weights.0.dot(&input);
        let layer1_relu = ActivationFunctions::relu2d(layer1);
        let layer2 = self.weights.1.dot(&layer1_relu);
        let output = ActivationFunctions::logsoftmax2d(layer2);
        let mut target_vec = vec![vec![0.0; 10]; target.len()];
        for (i, t) in target.iter().enumerate() {
            target_vec[i][*t as usize] = 1.0;
        }
        let target = Array2::from_shape_vec(
            (target_vec.len(), target_vec[0].len()),
            target_vec.into_iter().flatten().collect(),
        )
        .unwrap();
        let loss = -(&target * &output).sum();
        let layer2_gradients = ActivationFunctions::logsoftmax_backward2d(output, target);
        self.weights.1 = &self.weights.1 - &layer2_gradients * self.learning_rates.1;
        let layer1_gradients =
            ActivationFunctions::relu_backward2d(self.weights.1.dot(&input), layer2_gradients);
        self.weights.0 = &self.weights.0 - &layer1_gradients * self.learning_rates.0;
        loss
    }

    pub fn weights(&self) -> (Vec<f64>, Vec<f64>) {
        (
            self.weights.0.clone().into_raw_vec(),
            self.weights.1.clone().into_raw_vec(),
        )
    }
}
