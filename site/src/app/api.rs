use model::util::{Data, DataSingle, Weights};
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
