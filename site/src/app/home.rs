use crate::app::Grid;
use model::Model;
use yew::prelude::*;

#[function_component(Home)]
pub fn home() -> Html {
    let grid_component_handler = use_state(|| [[false; 28]; 28]);
    let inference_handler = use_state(|| 0);
    let show_grid_handle = use_state(|| false);

    let model = Model::new(
        (vec![vec![0.0; 728]; 128], vec![vec![0.0; 128]; 10]),
        (0.0, 0.0),
    );

    let infer_callback = {
        let inference_handler = inference_handler.clone();
        Callback::from(move |grid: [[bool; 28]; 28]| {
            let inference_handler = inference_handler.clone();
            let model = model.clone();
            wasm_bindgen_futures::spawn_local(async move {
                let grid_infer = grid
                    .iter()
                    .flatten()
                    .map(|x| if *x { 1.0 } else { 0.0 })
                    .collect::<Vec<f64>>();
                inference_handler.set(model.infer(grid_infer));
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
        </div>
    }
}
