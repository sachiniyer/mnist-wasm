# mnist-wasm

https://digits.sachiniyer.com

# Idea

1. Create a NN inside of the browser using wasm.
2. Load weights from an API, and allow people to update weights (by drawing on a canvas and labeling)
3. Post weights to API.
4. Build a training system completely in the browser using web workers to make it fast.
5. Use Rust for wasm, api, and model.

This is for fun to write an nn from scratch and also do some fun rust wasm stuff.

- [Information about the api](./api/README.md)
- [Information about the model](./model/README.md)
- [Information about the site](./site/README.md)

# Build/Run

- build/run both the api and site with `docker compose up --build` (beware of long build times).
- look at the `env.samples` to figure out what environment variables to set before running. Also the api generates weights on first run (if you don't copy them over into the image first).
