use axum::{
    http::header::{CONTENT_TYPE, USER_AGENT},
    http::Method,
    http::StatusCode,
    response::Html,
    routing::{delete, get, patch, post},
    Json, Router,
};
use csv;
use dotenv::dotenv;
use model;
use model::util;
use model::util::{get_sample_block, train_handler_wrapper, Data, DataInfo, DataSingle, Weights};
use serde_json::{json, Value};
use std::fs::File;
use std::sync::{Arc, Mutex};
use tower_http::cors::{Any, CorsLayer};

#[tokio::main]
async fn main() {
    dotenv().ok();
    check_envs();
    let shared_data = Arc::new(Mutex::new(Data { data: Vec::new() }));
    let weights = get_env("WEIGHTS");
    let bind_url = get_env("BIND_URL");

    if File::open(weights).is_err() {
        output_filter(format!("Creating weights file"), 0);
        let _ = weights_delete(shared_data.clone()).await;
    }
    let weights_delete_data = shared_data.clone();
    let sample_data_data = shared_data.clone();
    let sample_block_data = shared_data.clone();
    let cors = CorsLayer::new()
        .allow_methods([Method::GET, Method::POST, Method::PATCH, Method::DELETE])
        .allow_headers([USER_AGENT, CONTENT_TYPE])
        .allow_origin(Any);
    output_filter(format!("Listening on {}", bind_url), 0);
    axum::Server::bind(&bind_url.parse().unwrap())
        .serve(
            Router::new()
                .route("/", get(handler))
                .route(
                    "/weights",
                    delete(move || weights_delete(weights_delete_data)),
                )
                .route("/weights", get(weights_get))
                .route("/weights", post(weights_post))
                .route("/weights", patch(weights_patch))
                .route("/data", get(move || sample_data(sample_data_data)))
                .route(
                    "/datablock",
                    post(move |args| sample_data_block(args, sample_block_data)),
                )
                .layer(cors)
                .into_make_service(),
        )
        .await
        .unwrap();
}

async fn handler() -> &'static str {
    "This is the mnist-wasm api"
}

async fn weights_get() -> Json<Value> {
    let weights: Weights = get_weights();
    Json(json!(weights))
}

async fn weights_post(Json(weights): Json<Weights>) -> StatusCode {
    write_weights(&weights);
    StatusCode::OK
}

async fn sample_data(data: Arc<Mutex<Data>>) -> Json<Value> {
    data_refresh(data.clone());
    let data = data.lock().unwrap();
    let sample = Data {
        data: get_sample_block(&data, 1),
    };
    Json(json!(sample)).into()
}

async fn sample_data_block(Json(args): Json<DataInfo>, data: Arc<Mutex<Data>>) -> Json<Value> {
    data_refresh(data.clone());
    let data = data.lock().unwrap();
    let sample = Data {
        data: get_sample_block(&data, args.block),
    };
    Json(json!(sample)).into()
}

async fn weights_patch(Json(data): Json<Data>) -> Json<Value> {
    let lrate = get_env("LEARNING_RATE").parse::<f64>().unwrap();
    let weights: Weights = get_weights();
    let mut model = model::Model::new(weights.weights, (lrate, lrate));
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
            sync_weights(&model);
            Json(json!({ "loss": res }))
        }
    }
}

async fn weights_delete(data: Arc<Mutex<Data>>) -> Html<&'static str> {
    data_refresh(data.clone());

    let lrate = get_env("LEARNING_RATE").parse::<f64>().unwrap();
    let batch_size = get_env("BATCH_SIZE").parse::<usize>().unwrap();
    let iters = get_env("TRAIN_ITER").parse::<usize>().unwrap();

    let mut model = model::Model::new(
        (util::random_dist(784, 128), util::random_dist(128, 10)),
        (lrate, lrate),
    )
    .clone();

    output_filter(format!("Training for {} iterations", iters), 0);

    let mut iter = 0;
    while iter < iters {
        let data = data.lock().unwrap();
        let (loss, accuracy) = train_handler_wrapper(&data, &mut model, batch_size);
        output_filter(
            format!(
                "Iter {} -  Loss: {:.4} Accuracy {:.4}",
                iter, loss, accuracy
            ),
            1,
        );
        if iter % 500 == 0 {
            output_filter(format!("Iter {} - Syncing weights", iter), 0);
            sync_weights(&model);
        }
        iter += 1;
    }
    sync_weights(&model);
    output_filter(format!("Final Accuracy: {}", get_accuracy(&model)), 0);

    Html("Done")
}

fn check_envs() {
    let mut res = Vec::new();
    if std::env::var("WEIGHTS").is_err() {
        res.push("WEIGHTS ");
    }
    if std::env::var("BIND_URL").is_err() {
        res.push("BIND_URL ");
    }
    if std::env::var("DATA").is_err() {
        res.push("DATA ");
    }
    if std::env::var("LEARNING_RATE").is_err() {
        res.push("LEARNING_RATE");
    }
    if std::env::var("BATCH_SIZE").is_err() {
        res.push("BATCH_SIZE");
    }
    if std::env::var("TRAIN_ITER").is_err() {
        res.push("TRAIN_ITER");
    }
    if std::env::var("TEST_ITER").is_err() {
        res.push("TEST_ITER");
    }
    if std::env::var("OUTPUT_LEVEL").is_err() {
        res.push("OUTPUT_LEVEL");
    }
    if res.len() != 0 {
        println!("ENV VARS: {} are not set", res.join(", "));
        panic!("ENV VARS: {} are not set", res.join(", "));
    }
}

fn get_env(name: &str) -> String {
    std::env::var(name).unwrap()
}

fn output_filter(input: String, level: usize) {
    if get_env("OUTPUT_LEVEL").parse::<usize>().unwrap() > level {
        println!("{}", input);
    }
}

fn write_weights(weights: &Weights) {
    let temp = get_env("WEIGHTS") + ".tmp";
    if File::open(&temp).is_ok() {
        std::fs::remove_file(&temp).unwrap();
    }
    let mut temp_file = File::create(&temp).unwrap();
    serde_json::to_writer(&mut temp_file, &weights).unwrap();
    std::fs::rename(temp, get_env("WEIGHTS")).unwrap();
}

fn get_weights() -> Weights {
    let file = File::open(get_env("WEIGHTS")).unwrap();
    serde_json::from_reader(file).unwrap()
}

fn sync_weights(model: &model::Model) {
    let weights = Weights {
        weights: model.export_weights(),
    };
    write_weights(&weights);
}

fn data_refresh(data: Arc<Mutex<Data>>) {
    let mut data = data.lock().unwrap();
    if data.data.len() != 0 {
        return;
    }
    output_filter(format!("Loading training data"), 1);
    *data = read_data(
        format!("{}/xtrain.csv", get_env("DATA")),
        format!("{}/ytrain.csv", get_env("DATA")),
    );
}

fn read_data(xdata: String, ydata: String) -> Data {
    let mut xreader = csv::Reader::from_path(xdata).unwrap();
    let mut yreader = csv::Reader::from_path(ydata).unwrap();
    let mut xdata = Vec::new();
    let mut ydata = Vec::new();
    for (x, y) in xreader.records().zip(yreader.records()) {
        let x = x.unwrap();
        let y = y.unwrap();
        let mut xdata_single = Vec::new();
        for i in 0..x.len() {
            if x[i].parse::<f64>().unwrap() > 0.0 {
                xdata_single.push(1.0);
            } else {
                xdata_single.push(0.0);
            }
        }
        xdata.push(xdata_single);
        ydata.push(y[0].parse::<f64>().unwrap().round() as u8);
    }
    Data {
        data: xdata
            .into_iter()
            .zip(ydata.into_iter())
            .map(|(image, target)| DataSingle { image, target })
            .collect(),
    }
}

fn get_accuracy(model: &model::Model) -> f64 {
    let xdata = get_env("DATA") + "/xtest.csv";
    let ydata = get_env("DATA") + "/ytest.csv";
    let mut iters = get_env("TEST_ITER").parse::<usize>().unwrap();
    let batch_size = get_env("BATCH_SIZE").parse::<usize>().unwrap();
    let data = read_data(xdata, ydata);

    output_filter(format!("Testing for {} iterations", iters), 1);
    let mut accuracies = Vec::new();
    while iters > 0 {
        let chunk = get_sample_block(&data, batch_size);
        let (images, targets): (Vec<Vec<f64>>, Vec<u8>) =
            chunk
                .into_iter()
                .fold((Vec::new(), Vec::new()), |(mut images, mut targets), x| {
                    images.push(x.image.clone());
                    targets.push(x.target as u8);
                    (images, targets)
                });
        accuracies.push(
            model
                .infer2d(images.clone())
                .into_iter()
                .zip(targets.clone())
                .filter(|(x, y)| x == y)
                .count() as f64
                / batch_size as f64,
        );
        iters -= 1;
    }
    accuracies.iter().sum::<f64>() / accuracies.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn setup() {
        dotenv().ok();
        if File::open(get_env("WEIGHTS")).is_ok() {
            std::fs::remove_file(get_env("WEIGHTS")).unwrap();
        }
        let _ = File::create(get_env("WEIGHTS")).unwrap();
        let _ = weights_post(Json(Weights {
            weights: (util::random_dist(784, 128), util::random_dist(128, 10)),
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
