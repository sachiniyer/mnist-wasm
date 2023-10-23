use crate::api::get_block;
use js_sys::Uint8Array;
use model::util::Data;
use serde::{Deserialize, Serialize};
use wasm_bindgen::JsValue;
use yew_agent::prelude::*;
use yew_agent::Codec;

pub struct Postcard;

impl Codec for Postcard {
    fn encode<I>(input: I) -> JsValue
    where
        I: Serialize,
    {
        let data_json = serde_json::to_string(&input).expect("can't serialize a worker message");
        let data = data_json.as_bytes();
        let data = Uint8Array::from(data);
        JsValue::from(data)
    }

    fn decode<O>(input: JsValue) -> O
    where
        O: for<'de> Deserialize<'de>,
    {
        let data = Uint8Array::from(input);
        let data = data.to_vec();
        let data_json = String::from_utf8(data).expect("can't deserialize a worker message");
        serde_json::from_str(&data_json).expect("can't deserialize a worker message")
    }
}

#[oneshot]
pub async fn DataTask(n: usize) -> Data {
    get_block(n).await
}
