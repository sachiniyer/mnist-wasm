use model::util::{Data, DataInfo, DataSingle, Weights};
use reqwest::Client;

const API_URL: &str = "http://127.0.0.1:3000";

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

pub async fn get_block(block_size: usize) -> Data {
    let client = Client::new();
    let info = DataInfo { block: block_size };
    serde_json::from_str(
        &client
            .post(format!("{}/datablock", API_URL))
            .json(&info)
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap(),
    )
    .unwrap()
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
