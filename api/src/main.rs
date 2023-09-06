use axum::{response::Html, routing::get, Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
// use model;

#[derive(Serialize, Deserialize, Debug)]
struct Weights {
    weights: (Vec<Vec<f64>>, Vec<Vec<f64>>),
}

#[tokio::main]
async fn main() {
    let app = Router::new()
        .route("/", get(handler))
        .route("/refresh", get(refresh))
        .route("/weights", get(weights));

    axum::Server::bind(&"0.0.0.0:3000".parse().unwrap())
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

async fn weights() -> Json<Value> {
    Json(json!(Weights {
        weights: (
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec![vec![5.0, 6.0], vec![7.0, 8.0]]
        )
    }))
}
