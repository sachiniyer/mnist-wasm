use axum::{
    http::StatusCode,
    response::Html,
    routing::{get, post},
    Json, Router,
};
use dotenv::dotenv;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::fs::File;
use std::io::Write;
// use model;

#[derive(Serialize, Deserialize, Debug)]
struct Weights {
    weights: (Vec<Vec<f64>>, Vec<Vec<f64>>),
}

#[tokio::main]
async fn main() {
    dotenv().ok();
    if std::env::var("WEIGHTS").is_err() || std::env::var("BIND_URL").is_err() {
        return;
    }
    let app = Router::new()
        .route("/", get(handler))
        .route("/refresh", get(refresh))
        .route("/weights", get(weights_get))
        .route("/weights", post(weights_post));

    axum::Server::bind(&std::env::var("BIND_URL").unwrap().parse().unwrap())
        .serve(app.into_make_service())
        .await
        .unwrap();
}

async fn handler() -> Html<&'static str> {
    Html("<h1>Hello, World!</h1>")
}

async fn refresh() -> Html<&'static str> {
    Html("<h1>started weights refresh with mnist</h1>")
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
