use crate::model::Model;
use rand;
use rand::distributions::uniform;
use rand::seq::SliceRandom;
use rand::Rng;
use serde_derive::{Deserialize, Serialize};

pub fn random_dist(m: u32, h: u32) -> Vec<Vec<f64>> {
    let mut weights = Vec::new();
    for _ in 0..m {
        let mut row = Vec::new();
        for _ in 0..h {
            let mut rng = rand::thread_rng();
            row.push(rng.sample(uniform::Uniform::new(-1.0, 1.0)))
        }
        weights.push(row);
    }
    weights
}

pub fn random_int(m: u32, h: u32) -> Vec<Vec<u8>> {
    let mut weights = Vec::new();
    for _ in 0..m {
        let mut row = Vec::new();
        for _ in 0..h {
            let mut rng = rand::thread_rng();
            row.push((rng.sample(uniform::Uniform::new(0, 10)) as f64).round() as u8);
        }
        weights.push(row);
    }
    weights
}

pub fn approximate_equal(x: f64, y: f64, bound: Option<f64>) -> bool {
    match bound {
        Some(bound) => (x - y).abs() < bound,
        None => (x - y).abs() < 1e-4,
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Weights {
    pub weights: (Vec<Vec<f64>>, Vec<Vec<f64>>),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DataSingle {
    pub target: u8,
    pub image: Vec<f64>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Data {
    pub data: Vec<DataSingle>,
}

pub fn get_sample_block(data: &Data, size: usize) -> Vec<DataSingle> {
    let mut rng = rand::thread_rng();
    let mut data = data.clone();
    data.data.shuffle(&mut rng);
    data.data[0..size].to_vec()
}

pub fn train_handler(data: &Data, model: &mut Model, batch_size: usize) -> (f64, f64) {
    let chunk = get_sample_block(&data, batch_size);
    let (images, targets): (Vec<Vec<f64>>, Vec<u8>) =
        chunk
            .into_iter()
            .fold((Vec::new(), Vec::new()), |(mut images, mut targets), x| {
                images.push(x.image.clone());
                targets.push(x.target as u8);
                (images, targets)
            });
    let accuracy = model
        .infer2d(images.clone())
        .into_iter()
        .zip(targets.clone())
        .filter(|(x, y)| x == y)
        .count() as f64
        / batch_size as f64;

    let loss = model.train2d(images, targets);

    (loss, accuracy)
}
