use model::util::{Data, DataInfo, DataSingle, Weights};
use wasm_bindgen::JsValue;
use wasm_bindgen_futures::JsFuture;
use web_sys::{Request, RequestInit, RequestMode};
use reqwest::Client;

const API_URL: &str = "http://127.0.0.0:8000";

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

pub async fn get_block(block_size: usize) -> JsFuture {
    let mut opts = RequestInit::new();
    opts.method("POST");
    opts.mode(RequestMode::Cors);
    opts.body(Some(&JsValue::from_str(&serde_json::to_string(&DataInfo { block: block_size }).unwrap())));
    let request = Request::new_with_str_and_init(
        format!("{}/datablock", API_URL).as_str(),
        &opts,
    )
    .unwrap();
    request
        .headers()
        .set("Accept", "application/json")
        .unwrap();
    let window = web_sys::window().unwrap();
    // return a future that is fulfilled when the request is complete
    JsFuture::from(window.fetch_with_request(&request))
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
