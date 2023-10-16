use rand;
use rand::distributions::uniform;
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
