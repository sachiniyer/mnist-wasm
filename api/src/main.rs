use axum::{
    http::StatusCode,
    response::Html,
    routing::{delete, get, patch, post},
    Json, Router,
};
use dotenv::dotenv;
use model;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::fs::File;
use std::io::Write;

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Weights {
    weights: (Vec<Vec<f64>>, Vec<Vec<f64>>),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct DataSingle {
    target: u8,
    image: Vec<f64>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Data {
    data: Vec<DataSingle>,
}

#[tokio::main]
async fn main() {
    dotenv().ok();
    if std::env::var("WEIGHTS").is_err() || std::env::var("BIND_URL").is_err() {
        return;
    };
    axum::Server::bind(&std::env::var("BIND_URL").unwrap().parse().unwrap())
        .serve(app().into_make_service())
        .await
        .unwrap();
}
fn app() -> Router {
    Router::new()
        .route("/", get(handler))
        .route("/weights", delete(weights_delete))
        .route("/weights", get(weights_get))
        .route("/weights", post(weights_post))
        .route("/weights", patch(weights_patch))
}

async fn handler() -> &'static str {
    "This is the mnist-wasm api"
}

async fn weights_delete() -> Html<&'static str> {
    Html("<h1>started weights refresh with mnist</h1>")
}

async fn weights_patch(Json(data): Json<Data>) -> Json<Value> {
    let file = File::open(std::env::var("WEIGHTS").unwrap()).unwrap();
    let weights: Weights = serde_json::from_reader(file).unwrap();
    let mut model = model::Model::new(weights.weights, (0.1, 0.1));
    match data.data.len() {
        0 => Json(json!({"loss": 0})),
        1 => {
            let res = model.train1d(data.data[0].image.clone(), data.data[0].target.into());
            Json(json!({ "loss": res }))
        }
        _ => {
            let res = model.train2d(
                data.data
                    .clone()
                    .into_iter()
                    .map(|x| x.image.clone())
                    .collect(),
                data.data.into_iter().map(|x| x.target as i32).collect(),
            );

            Json(json!({ "loss": res }))
        }
    }
}

async fn weights_get() -> Json<Value> {
    let file = File::open(std::env::var("WEIGHTS").unwrap()).unwrap();
    let weights: Weights = serde_json::from_reader(file).unwrap();
    Json(json!({ "weights": weights.weights }))
}

async fn weights_post(Json(weights): Json<Weights>) -> StatusCode {
    let mut file = File::create(std::env::var("WEIGHTS").unwrap()).unwrap();
    file.write_all(serde_json::to_string(&weights).unwrap().as_bytes())
        .unwrap();
    StatusCode::OK
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approximate_equal(x: f64, y: f64) -> bool {
        (x - y).abs() < 1e-4
    }

    #[tokio::test]
    async fn test_handler() {
        let response = handler().await;
        assert_eq!(response, "This is the mnist-wasm api");
    }

    #[tokio::test]
    async fn test_weights_patch() {
        dotenv().ok();
        let _ = weights_post(Json(Weights {
            weights: (vec![vec![0.0; 784]; 128], vec![vec![0.0; 128]; 10]),
        }))
        .await;
        let response = weights_patch(Json(Data {
            data: vec![DataSingle {
                target: 1,
                image: vec![0.0; 784],
            }],
        }))
        .await;
        assert!(approximate_equal(
            response.0["loss"].as_f64().unwrap(),
            2.302585092994046
        ));
    }
}
