// Based off Yew Hooks crate https://github.com/jetli/yew-hooks/
use std::fmt;
use std::ops::Deref;
use std::rc::Rc;

use yew::prelude::*;

enum CounterAction {
    Increase,
    IncreaseBy(usize),
    Decrease,
    DecreaseBy(usize),
    Set(usize),
    Reset,
}

struct UseCounterReducer {
    value: usize,
    default: usize,
}

impl Reducible for UseCounterReducer {
    type Action = CounterAction;

    fn reduce(self: Rc<Self>, action: Self::Action) -> Rc<Self> {
        let next_value = match action {
            CounterAction::Increase => self.value + 1,
            CounterAction::IncreaseBy(delta) => self.value + delta,
            CounterAction::Decrease => {
                if self.value > 0 {
                    self.value - 1
                } else {
                    0
                }
            }
            CounterAction::DecreaseBy(delta) => {
                if self.value > delta {
                    self.value - delta
                } else {
                    0
                }
            }
            CounterAction::Set(value) => value,
            CounterAction::Reset => self.default,
        };

        Self {
            value: next_value,
            default: self.default,
        }
        .into()
    }
}

impl PartialEq for UseCounterReducer {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

// State handle for the [`use_counter`] hook.
pub struct UseCounterHandle {
    inner: UseReducerHandle<UseCounterReducer>,
}

impl UseCounterHandle {
    /// Increase by `1`.
    pub fn increase(&self) {
        self.inner.dispatch(CounterAction::Increase);
    }

    /// Increase by `delta`.
    pub fn increase_by(&self, delta: usize) {
        self.inner.dispatch(CounterAction::IncreaseBy(delta));
    }

    /// Decrease by `1`.
    pub fn decrease(&self) {
        self.inner.dispatch(CounterAction::Decrease);
    }

    /// Decrease by `delta`.
    pub fn decrease_by(&self, delta: usize) {
        self.inner.dispatch(CounterAction::DecreaseBy(delta));
    }

    /// Set to `value`.
    pub fn set(&self, value: usize) {
        self.inner.dispatch(CounterAction::Set(value));
    }

    /// Reset to initial value.
    pub fn reset(&self) {
        self.inner.dispatch(CounterAction::Reset);
    }
}

impl fmt::Debug for UseCounterHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UseCounterHandle")
            .field("value", &format!("{:?}", self.inner.value))
            .field("default", &format!("{:?}", self.inner.default))
            .finish()
    }
}

impl Deref for UseCounterHandle {
    type Target = usize;

    fn deref(&self) -> &Self::Target {
        &(*self.inner).value
    }
}

impl Clone for UseCounterHandle {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl PartialEq for UseCounterHandle {
    fn eq(&self, other: &Self) -> bool {
        *self.inner == *other.inner
    }
}

// This hook is used to manage counter state in a function component.
#[hook]
pub fn use_counter(default: usize) -> UseCounterHandle {
    let inner = use_reducer(move || UseCounterReducer {
        value: default,
        default,
    });

    UseCounterHandle { inner }
}
