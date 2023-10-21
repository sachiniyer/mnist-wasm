use site::data_agent::{DataTask, Postcard};
use yew_agent::Registrable;

fn main() {
    DataTask::registrar().encoding::<Postcard>().register();
}
