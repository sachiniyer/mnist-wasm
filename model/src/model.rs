use crate::activations::ActivationFunctions;
use ndarray::{Array1, Array2, Axis};

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

    pub fn export_weights(&self) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let mut res0 = Vec::new();
        let mut res1 = Vec::new();
        for i in self.weights.0.axis_iter(Axis(0)) {
            res0.push(i.clone().to_vec());
        }
        for i in self.weights.1.axis_iter(Axis(0)) {
            res1.push(i.clone().to_vec());
        }
        (res0, res1)
    }

    fn update_weights(&mut self, gradients: (Array2<f64>, Array2<f64>)) {
        self.weights.0 = &self.weights.0 - &gradients.0 * self.learning_rates.0;
        self.weights.1 = &self.weights.1 - &gradients.1 * self.learning_rates.1;
    }

    pub fn infer1d(&self, input: Vec<f64>) -> u8 {
        let input = Array1::from(input);
        let mut layer = input.dot(&self.weights.0);
        layer = ActivationFunctions::relu1d(layer);
        layer = layer.dot(&self.weights.1);
        layer
            .iter()
            .enumerate()
            .fold((0, 0.0), |(max_index, max_value), (index, value)| {
                if value > &max_value {
                    (index, *value)
                } else {
                    (max_index, max_value)
                }
            })
            .0 as u8
    }

    pub fn infer2d(&self, input: Vec<Vec<f64>>) -> Vec<u8> {
        let input = Array2::from_shape_vec(
            (input.len(), input[0].len()),
            input.into_iter().flatten().collect(),
        )
        .unwrap();
        let mut layer = input.dot(&self.weights.0);
        layer = ActivationFunctions::relu2d(layer);
        layer = layer.dot(&self.weights.1);
        layer
            .axis_iter(Axis(0))
            .map(|x| {
                x.iter()
                    .enumerate()
                    .fold((0, 0.0), |(max_index, max_value), (index, value)| {
                        if value > &max_value {
                            (index, *value)
                        } else {
                            (max_index, max_value)
                        }
                    })
                    .0 as u8
            })
            .collect()
    }

    pub fn train1d(&mut self, input: Vec<f64>, target: u8) -> f64 {
        let input = Array1::from(input);
        let layer1 = input.dot(&self.weights.0);
        let layer1_relu = ActivationFunctions::relu1d(layer1);
        let layer2 = layer1_relu.dot(&self.weights.1);
        let output = ActivationFunctions::logsoftmax1d(layer2);
        let mut target_vec = vec![0.0; 10];
        target_vec[target as usize] = 1.0;
        let target = Array1::from(target_vec);
        let loss = -(&target * &output).sum();
        let target_len = target.shape()[0];
        let target = -target / target_len as f64;
        let logsoftmax_gradients = ActivationFunctions::logsoftmax_backward1d(output, target);
        let layer2_gradients = layer1_relu
            .clone()
            .insert_axis(Axis(0))
            .t()
            .dot(&logsoftmax_gradients.clone().insert_axis(Axis(0)));
        let relu_gradients = ActivationFunctions::relu_backward1d(
            layer1_relu,
            logsoftmax_gradients.dot(&self.weights.1.t()),
        );
        let layer1_gradients = input
            .insert_axis(Axis(0))
            .t()
            .dot(&relu_gradients.insert_axis(Axis(0)));
        self.update_weights((layer1_gradients, layer2_gradients));
        loss
    }

    pub fn train2d(&mut self, input: Vec<Vec<f64>>, target: Vec<u8>) -> f64 {
        let input = Array2::from_shape_vec(
            (input.len(), input[0].len()),
            input.into_iter().flatten().collect(),
        )
        .unwrap();
        let layer1 = input.dot(&self.weights.0);
        let layer1_relu = ActivationFunctions::relu2d(layer1);
        let layer2 = layer1_relu.dot(&self.weights.1);
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
        let loss = (-(&target * &output)).mean_axis(Axis(1)).unwrap();
        let target_len = target.shape()[0];
        let target = -target / target_len as f64;
        let logsoftmax_gradients = ActivationFunctions::logsoftmax_backward2d(output, target);
        let layer2_gradients = layer1_relu.t().dot(&logsoftmax_gradients);
        let relu_gradients = ActivationFunctions::relu_backward2d(
            layer1_relu,
            logsoftmax_gradients.dot(&self.weights.1.t()),
        );
        let layer1_gradients = input.t().dot(&relu_gradients);
        self.update_weights((layer1_gradients, layer2_gradients));
        loss.sum() / target_len as f64
    }

    pub fn weights(&self) -> (Vec<f64>, Vec<f64>) {
        (
            self.weights.0.clone().into_raw_vec(),
            self.weights.1.clone().into_raw_vec(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_train1d() {
        let input = crate::util::random_dist(1, 784).get(0).unwrap().clone();
        let target = crate::util::random_int(1, 1)
            .get(0)
            .unwrap()
            .get(0)
            .unwrap()
            .clone();
        let mut model = Model::new(
            (
                crate::util::random_dist(784, 128),
                crate::util::random_dist(128, 10),
            ),
            (0.1, 0.1),
        );
        let loss = model.train1d(input, target);
        assert!(crate::util::approximate_equal(loss, 50.0, Some(100.0)))
    }

    #[test]
    fn test_train2d() {
        let input = crate::util::random_dist(128, 784);
        let target = crate::util::random_int(1, 128).get(0).unwrap().clone();
        let mut model = Model::new(
            (
                crate::util::random_dist(784, 128),
                crate::util::random_dist(128, 10),
            ),
            (0.1, 0.1),
        );
        let loss = model.train2d(input, target);
        assert!(crate::util::approximate_equal(loss, 6.0, Some(3.0)))
    }

    #[test]
    fn test_inference1d() {
        let input = crate::util::random_dist(1, 784).get(0).unwrap().clone();
        let model = Model::new(
            (
                crate::util::random_dist(784, 128),
                crate::util::random_dist(128, 10),
            ),
            (0.1, 0.1),
        );
        let prediction = model.infer1d(input);
        let targets: Vec<u8> = (0..10).collect();
        assert!(targets.contains(&prediction));
    }

    #[test]
    fn test_inference2d() {
        let input = crate::util::random_dist(256, 784);
        let model = Model::new(
            (
                crate::util::random_dist(784, 128),
                crate::util::random_dist(128, 10),
            ),
            (0.1, 0.1),
        );
        let prediction = model.infer2d(input);
        assert_eq!(prediction.len(), 256);
    }

    #[test]
    fn test_export() {
        let model = Model::new(
            (
                crate::util::random_dist(784, 128),
                crate::util::random_dist(128, 10),
            ),
            (0.1, 0.1),
        );
        let weights = model.export_weights();
        assert_eq!(weights.0.len(), 784);
        assert_eq!(weights.0[0].len(), 128);
        assert_eq!(weights.1.len(), 128);
        assert_eq!(weights.1[0].len(), 10);
    }
}
