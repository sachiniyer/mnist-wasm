FROM rust:latest as api-builder

WORKDIR /usr/src/app
COPY . .
RUN apt-get update && apt-get install -y wget unzip
RUN wget https://share.sachiniyer.com/api/public/dl/W0jDI6Qt -O pretrained.txt
RUN wget https://share.sachiniyer.com/api/public/dl/GlDsC5FY -O data.zip
RUN unzip data.zip
RUN cargo install --path ./api
CMD ["api"]

FROM debian:latest as api
WORKDIR /home/api
COPY --from=api-builder /usr/src/app/target/release/api .
COPY --from=api-builder /usr/src/app/pretrained.txt .
COPY --from=api-builder /usr/src/app/data ./data
EXPOSE 8000
CMD ["./api"]

FROM rust:latest as site-builder
WORKDIR /usr/src/app
COPY . .
RUN rustup target add wasm32-unknown-unknown
RUN apt-get update && apt-get install -y wget
RUN wget -qO- https://github.com/thedodd/trunk/releases/download/v0.17.5/trunk-x86_64-unknown-linux-gnu.tar.gz | tar -xzf-
RUN cargo install --locked wasm-bindgen-cli
RUN cd site && ../trunk build

FROM nginx:latest as site
WORKDIR /usr/share/nginx/html
COPY --from=site-builder /usr/src/app/site/dist .
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
