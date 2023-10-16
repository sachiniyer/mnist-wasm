use crate::app::api::get_weights;
use crate::app::Grid;
use model::util;
use model::Model;
use wasm_bindgen::JsCast;
use web_sys::{EventTarget, HtmlInputElement};
use yew::prelude::*;

#[function_component(Home)]
pub fn home() -> Html {
    let grid_component_handler = use_state(|| [[false; 28]; 28]);
    let inference_handler = use_state(|| 0);
    let show_grid_handle = use_state(|| false);
    let input_handle = use_state(|| 0);
    let loss_handle = use_state(|| 0.0);
    let model_handle = use_state(|| {
        Model::new(
            (util::random_dist(784, 128), util::random_dist(128, 10)),
            (0.0, 0.0),
        )
    });

    let infer_callback = {
        let inference_handler = inference_handler.clone();
        let model = model_handle.clone();
        Callback::from(move |grid: [[bool; 28]; 28]| {
            let inference_handler = inference_handler.clone();
            let model = model.clone();
            wasm_bindgen_futures::spawn_local(async move {
                let grid_infer = grid
                    .iter()
                    .flatten()
                    .map(|x| if *x { 1.0 } else { 0.0 })
                    .collect::<Vec<f64>>();
                inference_handler.set(model.infer1d(grid_infer));
            });
        })
    };

    let grid_callback = {
        let grid_component_handler = grid_component_handler.clone();
        Callback::from(move |grid: [[bool; 28]; 28]| {
            grid_component_handler.set(grid);
        })
    };

    let mod_callback = {
        Callback::from(move |grid: [[bool; 28]; 28]| {
            grid_callback.emit(grid);
            infer_callback.emit(grid);
        })
    };

    let show_grid_callback = {
        let show_grid_handle = show_grid_handle.clone();
        Callback::from(move |_| {
            show_grid_handle.set(!*show_grid_handle);
        })
    };

    fn print_grid(grid: [[bool; 28]; 28]) -> String {
        grid.iter()
            .map(|row| {
                row.iter()
                    .map(|col| if *col { "1" } else { "0" })
                    .collect::<Vec<&str>>()
                    .join("")
            })
            .collect::<Vec<String>>()
            .join("\n")
    }

    let train_callback = {
        let input_handle = input_handle.clone();
        let grid_component_handler = grid_component_handler.clone();
        let loss_handle = loss_handle.clone();
        let model_handle = model_handle.clone();
        Callback::from(move |_| {
            let input = (*input_handle).clone();
            let grid = (*grid_component_handler).clone();
            let loss_handle = loss_handle.clone();
            let model_handle = model_handle.clone();
            wasm_bindgen_futures::spawn_local(async move {
                let mut model = (*model_handle).clone();
                let grid_train = grid
                    .iter()
                    .flatten()
                    .map(|x| if *x { 1.0 } else { 0.0 })
                    .collect::<Vec<f64>>();
                let loss = model.train1d(grid_train, input);
                loss_handle.set(loss);
                model_handle.set(model);
            });
        })
    };

    let input_callback = {
        let input_handle = input_handle.clone();
        Callback::from(move |e: Event| {
            let target: Option<EventTarget> = e.target();
            let input = target.and_then(|t| t.dyn_into::<HtmlInputElement>().ok());
            if let Some(input) = input {
                input_handle.set(input.value().parse::<u8>().unwrap());
            }
        })
    };

    let weights_callback = {
        let model_handle = model_handle.clone();
        Callback::from(move |_| {
            let model_handle = model_handle.clone();
            wasm_bindgen_futures::spawn_local(async move {
                let weights = get_weights().await;
                let new_model = Model::new(weights.weights, (0.0, 0.0));
                model_handle.set(new_model);
                web_sys::window()
                    .unwrap()
                    .alert_with_message("Weights loaded from API")
                    .unwrap();
            });
        })
    };

    html! {
        <div>
            <h1>{ "MNIST WASM" }</h1>
            <p>{ "rust wasm neural net in your browser" }</p>
            <div>
                <Grid grid={ mod_callback }
                      init_grid={ [[false; 28]; 28] }/>
            </div>
            <button onclick={ show_grid_callback }>{ "Show Data" }</button>
            <div> {
                if *show_grid_handle { print_grid(*grid_component_handler) }
                else { "".to_string() }
            } </div>
            <div>{ format!("Inference: {}", *inference_handler) }</div>
            <div>
                <button onclick={ train_callback }>{ "Train Model" }</button>
                <input onchange={ input_callback }type="number" id="target" name="target" min="0" max="9" />
                <div>{ format!("Loss {}", *loss_handle) }</div>
            </div>
            <div>
                <button onclick={ weights_callback }>{ "Load weights from API" }</button>
            </div>
        </div>
    }
}
