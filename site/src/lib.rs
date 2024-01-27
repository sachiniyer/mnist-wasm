use wasm_logger::{init, Config};
use yew::prelude::*;
use yew_agent::reactor::ReactorProvider;

pub mod api;
pub mod grid;
pub mod home;

use grid::Grid;
use home::Home;

pub mod model_agent;
use model_agent::ModelReactor;

#[function_component]
pub fn App() -> Html {
    init(Config::default());
    html! {
            <ReactorProvider<ModelReactor> path="/worker_model.js">
                <Home />
            </ReactorProvider<ModelReactor>>
    }
}
