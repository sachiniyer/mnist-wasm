use model::util::{Data, DataInfo, DataSingle, Weights};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsValue;
use wasm_bindgen_futures::JsFuture;
use web_sys::{Request, RequestInit, RequestMode, Response};
use reqwest::Client;

const API_URL: &str = "http://127.0.0.0:8000";

pub struct Sendable<T: ?Sized>(pub Box<T>);

pub async fn get_weights() -> Weights {
    let client = Client::new();
    serde_json::from_str(
        &client
            .get(format!("{}/weights", API_URL))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap(),
    )
    .unwrap()
}

pub async fn get_sample() -> DataSingle {
    let client = Client::new();
    let data: Data = serde_json::from_str(
        &client
            .get(format!("{}/data", API_URL))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap(),
    )
    .unwrap();
    data.data.get(0).unwrap().clone()
}

pub async fn get_block(block_size: usize) -> Sendable<Result<JsValue, JsValue>> {
    let data_info = DataInfo { block: block_size };
    let serialized_data_info = serde_json::to_string(&data_info).map_err(|e| JsValue::from_str(&e.to_string())).unwrap();
    // let mut opts = RequestInit::new();
    // opts.method("POST");
    // opts.mode(RequestMode::Cors);
    // opts.body(Some(&JsValue::from_str(&serialized_data_info)));

    let request = Request::new(Method::POST, &format!("{}/datablock", API_URL))
        .mode(RequestMode::Cors)
        .header("Content-Type", "application/json")
        .header("Accept", "application/json");

    let window = web_sys::window().unwrap();
}

pub async fn send_weights(weights: Weights) {
    let client = Client::new();
    client
        .post(format!("{}/weights", API_URL))
        .json(&weights)
        .send()
        .await
        .unwrap();
}

pub async fn weights_delete() {
    let client = Client::new();
    client
        .delete(format!("{}/weights", API_URL))
        .send()
        .await
        .unwrap();
}
