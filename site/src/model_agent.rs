use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use yew_agent::prelude::*;

use model::util::{random_dist, train_handler_wrapper, Data, Weights};
use model::Model;

use std::collections::VecDeque;

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ControlSignal {
    Start,
    Stop,
    GetWeights,
    SetWeights(Weights),
    SetBatchSize(usize),
    SetLearningRate(i64),
    AddData(Data),
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ResponseSignal {
    weights: Weights,
    loss: f64,
    acc: f64,
    batch_size: usize,
    lrate: i64,
    data_len: usize,
}

#[reactor]
pub async fn ModelReactor(mut scope: ReactorScope<ControlSignal, ResponseSignal>) {
    let mut data_vec = VecDeque::new();
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
            ControlSignal::GetWeights => {}
            ControlSignal::SetWeights(w) => {
                model = Model::new(w.weights, (lrate, lrate));
            }
            ControlSignal::SetBatchSize(b) => {
                batch_size = b;
            }
            ControlSignal::SetLearningRate(l) => {
                let l = l as f64;
                model = Model::new(model.export_weights(), (l, l));
            }
            ControlSignal::AddData(d) => {
                data_vec.push_front(d);
            }
        };
        if training {
            if !data_vec.is_empty() {
                let mut loss = 0.0;
                let mut acc = 0.0;
                if data_vec.front().unwrap().data.len() == batch_size {
                    (loss, acc) = train_handler_wrapper(
                        &data_vec.pop_front().unwrap(),
                        &mut model,
                        batch_size,
                    );
                }
                scope
                    .send(ResponseSignal {
                        weights: Weights {
                            weights: model.export_weights(),
                        },
                        loss,
                        acc,
                        batch_size,
                        lrate: lrate as i64,
                        data_len: data_vec.len(),
                    })
                    .await
                    .unwrap();
            }
        };
    }
}
