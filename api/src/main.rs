use axum::{
    http::StatusCode,
    response::Html,
    routing::{delete, get, patch, post},
    Json, Router,
};
use csv;
use dotenv::dotenv;
use model;
use model::util;
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
    if std::env::var("WEIGHTS").is_err()
        || std::env::var("BIND_URL").is_err()
        || std::env::var("DATA").is_err()
    {
        let mut res = String::from("ENV vars not set: ");
        if std::env::var("WEIGHTS").is_err() {
            res.push_str("WEIGHTS ");
        }
        if std::env::var("BIND_URL").is_err() {
            res.push_str("BIND_URL ");
        }
        if std::env::var("DATA").is_err() {
            res.push_str("DATA ");
        }
        println!("{}", res);
        return;
    };
    if File::open(std::env::var("WEIGHTS").unwrap()).is_err() {
        println!("Creating weights file");
        let _ = weights_delete().await;
    }
    // axum::Server::bind(&std::env::var("BIND_URL").unwrap().parse().unwrap())
    //     .serve(app().into_make_service())
    //     .await
    //     .unwrap();
    println!("Listening on {}", std::env::var("BIND_URL").unwrap());
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
    let xdata = std::env::var("DATA").unwrap() + "/xtrain.csv";
    let ydata = std::env::var("DATA").unwrap() + "/ytrain.csv";
    let mut xreader = csv::Reader::from_path(xdata).unwrap();
    let mut yreader = csv::Reader::from_path(ydata).unwrap();
    let mut xdata = Vec::new();
    let mut ydata = Vec::new();
    for (x, y) in xreader.records().zip(yreader.records()) {
        let x = x.unwrap();
        let y = y.unwrap();
        let mut xdata_single = Vec::new();
        for i in 0..x.len() {
            // if x[i].parse::<f64>().unwrap() > 0.0 {
            //     xdata_single.push(1.0);
            // } else {
            //     xdata_single.push(0.0);
            // }
            xdata_single.push(x[i].parse::<f64>().unwrap());
        }
        xdata.push(xdata_single);
        ydata.push(y[0].parse::<f64>().unwrap().round() as u8);
    }
    if File::open(std::env::var("WEIGHTS").unwrap()).is_ok() {
        std::fs::remove_file(std::env::var("WEIGHTS").unwrap()).unwrap();
    }
    let mut file = File::create(std::env::var("WEIGHTS").unwrap()).unwrap();
    let data = Data {
        data: xdata
            .into_iter()
            .zip(ydata.into_iter())
            .map(|(image, target)| DataSingle { image, target })
            .collect(),
    };
    let mut model = model::Model::new(
        (util::random_dist(784, 128), util::random_dist(128, 10)),
        (0.001, 0.001),
    )
    .clone();
    let batch_size = 128;
    for chunk in data.data.chunks(batch_size).into_iter() {
        if chunk.len() != batch_size {
            break;
        }
        let loss = model.train2d(
            chunk.clone().into_iter().map(|x| x.image.clone()).collect(),
            chunk.into_iter().map(|x| x.target as u8).collect(),
        );
        let accuracy = chunk.into_iter().fold(0, |acc, x| {
            if model.infer(x.image.clone()) == x.target {
                acc + 1
            } else {
                acc
            }
        }) as f64
            / batch_size as f64;
        println!("Loss: {}", loss);
        println!("Accuracy: {}", accuracy);
        let weights = model.export_weights();
        file.write_all(
            serde_json::to_string(&Weights { weights })
                .unwrap()
                .as_bytes(),
        )
        .unwrap();
    }
    Html("Done")
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
                data.data.into_iter().map(|x| x.target as u8).collect(),
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

    async fn setup() {
        dotenv().ok();
        if File::open(std::env::var("WEIGHTS").unwrap()).is_ok() {
            std::fs::remove_file(std::env::var("WEIGHTS").unwrap()).unwrap();
        }
        let _ = File::create(std::env::var("WEIGHTS").unwrap()).unwrap();
        let _ = weights_post(Json(Weights {
            weights: (vec![vec![0.0; 784]; 128], vec![vec![0.0; 128]; 10]),
        }))
        .await;
    }

    #[tokio::test]
    async fn test_handler() {
        let response = handler().await;
        assert_eq!(response, "This is the mnist-wasm api");
    }

    #[tokio::test]
    async fn test_weights_patch() {
        setup().await;
        let response = weights_patch(Json(Data {
            data: vec![DataSingle {
                target: 1,
                image: vec![0.0; 784],
            }],
        }))
        .await;
        assert!(util::approximate_equal(
            response.0["loss"].as_f64().unwrap(),
            2.30258509,
            None
        ));
    }
}
