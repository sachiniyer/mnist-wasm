# Model

Model for mnist-wasm

## Design

Layers

- 128 relu
- 10 logsoftmax

I kept the model super simple, because I want it to run fast in the browser. I also use logsoftmax because I was afraid that I was going to have overflow issues. It also gave me much better results when prototyping in python.

Everything is implemented in scratch with rust. The only significant library used is [ndarray](https://docs.rs/ndarray/latest/ndarray/) to make matrix calculations a bit easier
