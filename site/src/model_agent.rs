use std::time::Duration;

use futures::sink::SinkExt;
use futures::{FutureExt, StreamExt};
use serde::{Deserialize, Serialize};
use yew::platform::time::sleep;
use yew_agent::prelude::*;

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ControlSignal {
    Start,
    Stop,
}

#[reactor]
pub async fn ModelReactor(mut scope: ReactorScope<ControlSignal, u64>) {
    let mut hold_num = 1;
    while let Some(m) = scope.next().await {
        if m == ControlSignal::Start {
            'inner: for i in hold_num.. {
                // This is not the most efficient way to calculate prime,
                // but this example is here to demonstrate how primes can be
                // sent to the application in an ascending order.
                if primes::is_prime(i) {
                    scope.send(i).await.unwrap();
                }

                futures::select! {
                    m = scope.next() => {
                        if m == Some(ControlSignal::Stop) {
                            hold_num = i;
                            break 'inner;
                        }
                    },
                    _ = sleep(Duration::from_millis(100)).fuse() => {},
                }
            }
        }
    }
}
