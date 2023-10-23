use futures::sink::SinkExt;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use yew_agent::oneshot::use_oneshot_runner;
use yew_agent::prelude::*;

use crate::data_agent::DataTask;
use model::util::random_dist;
use model::util::Weights;
use model::Model;

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ControlSignal {
    Start,
    Stop,
    GetWeights,
    SetWeights { weights: Weights },
    SetBatchSize { batch_size: usize },
    SetLearningRate { learning_rate: i64 },
}

#[reactor]
pub async fn ModelReactor(mut scope: ReactorScope<ControlSignal, Weights>) {
    let data_task = use_oneshot_runner::<DataTask>();
    let mut training = false;
    let mut batch_size = 128;
    let lrate = 0.01;
    let mut model = Model::new(
        (random_dist(784, 128), random_dist(128, 10)),
        (lrate, lrate),
    );
    while let Some(m) = scope.next().await {
        match m {
            ControlSignal::Start => {
                training = true;
            }
            ControlSignal::Stop => {
                training = false;
            }
            ControlSignal::GetWeights => {
                scope
                    .send(Weights {
                        weights: model.export_weights(),
                    })
                    .await
                    .unwrap();
            }
            ControlSignal::SetWeights { weights: w } => {
                model = Model::new(w.weights, (lrate, lrate));
            }
            ControlSignal::SetBatchSize { batch_size: b } => {
                batch_size = b;
            }
            ControlSignal::SetLearningRate { learning_rate: l } => {
                let l = l as f64;
                model = Model::new(model.export_weights(), (l, l));
            }
        }
    }
}
