use futures::{FutureExt, SinkExt, StreamExt};
use model::{
    util::{random_dist, train_handler_wrapper, Data, Weights},
    Model,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
    time::Duration,
};
use yew::platform::time::sleep;
use yew_agent::prelude::*;

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ControlSignal {
    Start,
    Stop,
    GetStatus,
    SetWeights(Weights),
    SetBatchSize(usize),
    SetLearningRate(i64),
    AddData(Data),
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ResponseSignal {
    pub weights: Weights,
    pub loss: f64,
    pub acc: f64,
    pub batch_size: usize,
    pub lrate: f64,
    pub data_len: usize,
    pub iteration: usize,
}

#[reactor]
pub async fn ModelReactor(mut scope: ReactorScope<ControlSignal, ResponseSignal>) {
    web_sys::console::log_1(&"Model agent started".into());
    async fn respond(
        scope: &mut ReactorScope<ControlSignal, ResponseSignal>,
        weights: (Vec<Vec<f64>>, Vec<Vec<f64>>),
        loss: f64,
        acc: f64,
        batch_size: usize,
        lrate: f64,
        iteration: usize,
        data: &VecDeque<Data>,
    ) {
        scope
            .send(ResponseSignal {
                weights: Weights { weights },
                loss,
                acc,
                batch_size,
                lrate,
                data_len: data.len(),
                iteration,
            })
            .await
            .unwrap();
    }

    let data_vec: Arc<Mutex<VecDeque<Data>>> = Arc::new(Mutex::new(VecDeque::new()));
    let mut training = false;
    let mut batch_size: usize = 128;
    let lrate: f64 = 0.01;
    let mut loss = 0.0;
    let mut acc = 0.0;
    let mut iteration = 0;
    let mut model = Model::new(
        (random_dist(784, 128), random_dist(128, 10)),
        (lrate, lrate),
    );
    let mut send_status = true;

    loop {
        if data_vec.lock().unwrap().len() != batch_size {
            data_vec.lock().unwrap().clear();
        }

        if send_status {
            respond(
                &mut scope,
                model.export_weights(),
                loss,
                acc,
                batch_size,
                lrate,
                iteration,
                &data_vec.clone().lock().unwrap(),
            )
            .await;
            send_status = false;
        }

        if training {
            if !data_vec.lock().unwrap().is_empty() {
                (loss, acc) = train_handler_wrapper(
                    &data_vec.lock().unwrap().pop_front().unwrap(),
                    &mut model,
                    batch_size,
                );
                iteration += 1;
            }
            send_status = true;
        }
        futures::select! {
            c = scope.next() => {
                if let Some(c) = c {
                    match c {
                        ControlSignal::Start => {
                            web_sys::console::log_1(&"Starting training".into());
                            iteration = 0;
                            training = true;
                        }
                        ControlSignal::Stop => {
                            web_sys::console::log_1(&"Stopping training".into());
                            training = false;
                        }
                        ControlSignal::GetStatus => {
                            web_sys::console::log_1(&"Sending status".into());
                            send_status = true;
                        }
                        ControlSignal::SetWeights(w) => {
                            web_sys::console::log_1(&"Setting weights".into());
                            model = Model::new(w.weights, (lrate, lrate));
                        }
                        ControlSignal::SetBatchSize(b) => {
                            web_sys::console::log_1(&"Setting batch size".into());
                            batch_size = b;
                        }
                        ControlSignal::SetLearningRate(l) => {
                            web_sys::console::log_1(&"Setting learning rate".into());
                            let l = l as f64;
                            model = Model::new(model.export_weights(), (l, l));
                        }
                        ControlSignal::AddData(d) => {
                            web_sys::console::log_1(&"Adding data".into());
                            data_vec.lock().unwrap().push_back(d);
                        }
                    };
                } else {
                    continue;
                }
            }
            _ = sleep(Duration::from_millis(100)).fuse() => {
                continue;
            }
        };
    }
}
