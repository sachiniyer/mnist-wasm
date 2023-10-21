FROM rust:latest as api-builder

WORKDIR /usr/src/app
COPY . .
RUN apt-get update && apt-get install -y wget
RUN wget https://share.sachiniyer.com/api/public/dl/W0jDI6Qt -O pretrained.txt
RUN cargo install --path ./api
CMD ["api"]

FROM debian:latest as api
WORKDIR /home/api
COPY --from=api-builder /usr/src/app/target/release/api .
COPY --from=api-builder /usr/src/app/pretrained.txt .
# COPY ./data ./data
CMD ["./api"]

FROM rust:latest as site-builder
WORKDIR /usr/src/app
COPY . .
RUN rustup target add wasm32-unknown-unknown
RUN apt-get update && apt-get install -y wget
RUN wget -qO- https://github.com/thedodd/trunk/releases/download/v0.17.5/trunk-x86_64-unknown-linux-gnu.tar.gz | tar -xzf-
RUN cargo install --locked wasm-bindgen-cli
RUN cd site && ../trunk build

FROM busybox:latest as site
RUN adduser -D -u 1000 site
USER site
WORKDIR /home/site
COPY --from=site-builder /usr/src/app/site/dist .
CMD ["busybox", "httpd", "-f", "-p", "3000", "-h", "/home/site"]
