# API

API for mnist-wasm

## Design

The API is intentionally very simple

### Routes

#### GET /

Just a `hello world`

#### GET /refresh

Clears and refreshes with weights with the [mnist dataset](http://yann.lecun.com/exdb/lenet/index.html)

#### GET /weights

Just responds with the weights that are stored locally

#### POST /weights

Updates the weights when someones decides to upload them
