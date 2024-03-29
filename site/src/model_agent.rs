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
use crate::api::get_block;
use wasm_bindgen_futures::spawn_local;

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ControlSignal {
    Start,
    Stop,
    GetStatus,
    SetWeights(Weights),
    SetBatchSize(usize),
    SetLearningRate(i64),
    SetCacheSize(usize),
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ResponseSignal {
    pub weights: Weights,
    pub loss: f64,
    pub acc: f64,
    pub batch_size: usize,
    pub lrate: f64,
    pub data_len: usize,
    pub data_futures_len: usize,
    pub iteration: usize,
    pub cache_size: usize,
}

pub struct ModelData {
    data_vec: Arc<Mutex<VecDeque<Data>>>,
    data_given: i64,
    data_taken: i64,
    training: bool,
    batch_size: usize,
    lrate: f64,
    loss: f64,
    acc: f64,
    iteration: usize,
    cache_size: usize,
    model: Model,
    send_status: bool,

}

impl ModelData {
    fn new() -> Self {
        Self {
            data_vec: Arc::new(Mutex::new(VecDeque::new())),
            data_given: 0,
            data_taken: 0,
            training: false,
            batch_size: 128,
            lrate: 0.01,
            loss: 0.0,
            acc: 0.0,
            iteration: 0,
            cache_size: 5,
            model: Model::new(
                (random_dist(784, 128), random_dist(128, 10)),
                (0.01, 0.01),
            ),
            send_status: true,
        }
    }

    async fn execute(&mut self) {
        if self.training {
            self.train();
        }
        self.cache_data().await;
    }

    fn respond(&mut self) -> ResponseSignal {
        ResponseSignal {
            weights: Weights{ weights: self.model.export_weights()},
            loss: self.loss,
            acc: self.acc,
            batch_size: self.batch_size,
            lrate: self.lrate,
            data_len: self.data_vec.lock().unwrap().len(),
            data_futures_len: (self.data_given - self.data_taken) as usize,
            iteration: self.iteration,
            cache_size: self.cache_size,
        }
    }

    fn train(&mut self) {
        let mut data = self.data_vec.lock().unwrap().pop_front();
        while data.is_some() && data.as_ref().unwrap().data.len() != self.batch_size {
            data = self.data_vec.lock().unwrap().pop_front();
            self.data_taken += 1;
        }
        if data.is_some() {
            self.data_taken += 1;
            let (loss, acc) = train_handler_wrapper(&data.unwrap(), &mut self.model, self.batch_size);
            self.loss = loss;
            self.acc = acc;
            self.iteration += 1;
            self.send_status = true;
        }
    }

    async fn cache_data(&mut self) {
        let missing = (self.cache_size as i64 - self.data_vec.lock().unwrap().len() as i64) - (self.data_given - self.data_taken);
        self.add_futures(missing).await;
    }

    async fn add_futures(&mut self, missing: i64) {
        for _ in 0..missing {
            self.data_given += 1;
            let data_vec_handle = self.data_vec.clone();
            let batch_size = self.batch_size;
            spawn_local(async move {
                let data = get_block(batch_size).await;
                data_vec_handle.lock().unwrap().push_back(data);
            });
        }
    }

    fn set_send_status(&mut self, status: bool) {
        self.send_status = status;
    }

    fn set_training(&mut self, status: bool) {
        self.training = status;
        if status {
            self.iteration = 0;
        }
    }

    fn set_weights(&mut self, weights: Weights) {
        self.model = Model::new(weights.weights, (self.lrate, self.lrate));
    }

    fn set_batch_size(&mut self, batch_size: usize) {
        self.batch_size = batch_size;
    }

    fn set_learning_rate(&mut self, lrate: f64) {
        self.lrate = lrate;
        self.model = Model::new(self.model.export_weights(), (lrate, lrate));
    }

    fn set_cache_size(&mut self, cache_size: usize) {
        self.cache_size = cache_size;
    }

}

#[reactor]
#[allow(non_snake_case)]
pub async fn ModelReactor(mut scope: ReactorScope<ControlSignal, ResponseSignal>) {
    let mut data = ModelData::new();
    loop {
        data.execute().await;
        if data.send_status {
            scope.send(data.respond()).await.unwrap();
        }
        futures::select! {
            c = scope.next() => {
                if let Some(c) = c {
                    match c {
                        ControlSignal::Start => {
                            web_sys::console::log_1(&"Starting training".into());
                            data.set_training(true);
                        }
                        ControlSignal::Stop => {
                            web_sys::console::log_1(&"Stopping training".into());
                            data.set_training(false);
                        }
                        ControlSignal::GetStatus => {
                            web_sys::console::log_1(&"Sending status".into());
                            data.set_send_status(true);
                        }
                        ControlSignal::SetWeights(w) => {
                            web_sys::console::log_1(&"Setting weights".into());
                            data.set_weights(w);
                        }
                        ControlSignal::SetBatchSize(b) => {
                            web_sys::console::log_1(&"Setting batch size".into());
                            data.set_batch_size(b);
                        }
                        ControlSignal::SetLearningRate(l) => {
                            web_sys::console::log_1(&"Setting learning rate".into());
                            data.set_learning_rate(l as f64);
                        }
                        ControlSignal::SetCacheSize(c) => {
                            web_sys::console::log_1(&"Setting cache size".into());
                            data.set_cache_size(c);
                        }
                    };
                } else {
                    continue;
                }
            }
            _ = sleep(Duration::from_millis(10)).fuse() => {
                continue;
            }
        };
    }
}
