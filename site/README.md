# Site

Frontend for mnist-wasm

## Design

Built with [yew](https://yew.rs)

Features:

- In-browser inference done asynchronously while drawing
- In-browser training
- API interaction for weight upload and download (everything else is done in your browser)
- Cool grid to draw your characters

This is built using rust to create wasm that both runs the model and creates the website. This allows for all training and inference to be done at close to native speed.

The weights are the only thing that are pushed/pulled from the API (so you don't have to wait for the model to train every time). It also allows anyone to improve the weights when they visit the site for the next people.
