use yew::prelude::*;
use yew_agent::{oneshot::OneshotProvider, reactor::ReactorProvider};
pub mod api;
pub mod grid;
pub mod home;

use grid::Grid;
use home::Home;

pub mod model_agent;
use model_agent::ModelReactor;
pub mod data_agent;
use data_agent::{DataTask, Postcard};

#[function_component]
pub fn App() -> Html {
    wasm_logger::init(wasm_logger::Config::default());
    html! {
        <OneshotProvider<DataTask, Postcard> path="/data_worker.js">
            <ReactorProvider<ModelReactor> path="/model_worker.js">
                <Home />
            </ReactorProvider<ModelReactor>>
        </OneshotProvider<DataTask, Postcard>>
    }
}
