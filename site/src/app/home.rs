use crate::app::api::{get_weights, send_weights, weights_delete};
use crate::app::Grid;
use model::util;
use model::util::Weights;
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
    let model_handle_effect = model_handle.clone();
    use_effect(move || {
        let model = model_handle_effect.clone();
        wasm_bindgen_futures::spawn_local(async move {
            let weights = get_weights().await;
            let new_model = Model::new(weights.weights, (0.0, 0.0));
            model.set(new_model);
        });
        || {}
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

    let load_weights_callback = {
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

    let send_weights_callback = {
        let model_handle = model_handle.clone();
        Callback::from(move |_| {
            let model_handle = model_handle.clone();
            wasm_bindgen_futures::spawn_local(async move {
                let model = (*model_handle).clone();
                let weights = model.export_weights();
                send_weights(Weights { weights }).await;
                web_sys::window()
                    .unwrap()
                    .alert_with_message("Weights sent to API")
                    .unwrap();
            });
        })
    };

    let delete_weights_callback = {
        let model_handle = model_handle.clone();
        Callback::from(move |_| {
            let model_handle = model_handle.clone();
            wasm_bindgen_futures::spawn_local(async move {
                weights_delete().await;
                let weights = get_weights().await;
                let new_model = Model::new(weights.weights, (0.0, 0.0));
                model_handle.set(new_model);
                web_sys::window()
                    .unwrap()
                    .alert_with_message("Weights deleted from API")
                    .unwrap();
            });
        })
    };

    html! {
        <div>
            <div>
                <h1>{ "MNIST WASM" }</h1>
                <div id="info">
                    <p>{ "rust wasm neural net in your browser" }</p>
                    <a href="https://github.com/sachiniyer/mnist-wasm">{ "source" }</a>
                </div>
            </div>
            <div id="wrapper">
                <div id="left">
                    <Grid grid={ mod_callback }
                          init_grid={ [[false; 28]; 28] }/>
                    <button class="grid-control" onclick={ show_grid_callback }>{ "Show Data" }</button>
                    <div> {
                        if *show_grid_handle { print_grid(*grid_component_handler) }
                        else { "".to_string() }
                    } </div>
                </div>
                <div id="right">
                    <div id="weights">
                        <div>
                            <button onclick={ load_weights_callback }>{ "Reset weights with API" }</button>
                        </div>
                        <div>
                            <button onclick={ send_weights_callback }>{ "Send weights to API" }</button>
                        </div>
                        <div>
                            <button onclick={ delete_weights_callback }>{ "Delete weights in API" }</button>
                        </div>
                    </div>
                    <div><p id="inference">{ format!("Inference: {}", *inference_handler) }</p></div>
                    <div >
                        <button id="tune" onclick={ train_callback }>{ "Tune Model" }</button>
                        <div id="loss-div">
                            <input onchange={ input_callback }type="number" id="target" name="target" min="0" max="9" />
                            <p id="loss">{ format!("Loss: {}", *loss_handle) }</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    }
}
