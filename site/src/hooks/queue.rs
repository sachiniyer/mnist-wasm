// Based off Yew Hooks crate https://github.com/jetli/yew-hooks/
use std::cell::{Ref, RefCell};
use std::collections::VecDeque;
use std::hash::Hash;
use std::rc::Rc;

use yew::prelude::*;

#[hook]
pub fn use_update() -> Rc<dyn Fn()> {
    let state = use_state(|| 0);

    Rc::new(move || {
        state.set((*state + 1) % 1_000_000);
    })
}
// State handle for the [`use_queue`] hook.
pub struct UseQueueHandle<T> {
    inner: Rc<RefCell<VecDeque<T>>>,
    update: Rc<dyn Fn()>,
}

impl<T> UseQueueHandle<T> {
    /// Get immutable ref to the queue.
    ///
    /// # Panics
    ///
    /// Panics if the value is currently mutably borrowed
    pub fn current(&self) -> Ref<VecDeque<T>> {
        self.inner.borrow()
    }

    /// Set the queue.
    pub fn set(&self, queue: VecDeque<T>) {
        *self.inner.borrow_mut() = queue;
        (self.update)();
    }

    /// Appends an element to the back of the queue.
    pub fn push_back(&self, value: T)
    where
        T: Eq + Hash,
    {
        self.inner.borrow_mut().push_back(value);
        (self.update)();
    }

    /// Removes the first element and returns it, or None if the queue is empty.
    pub fn pop_front(&self) -> Option<T>
    where
        T: Eq + Hash,
    {
        let v = self.inner.borrow_mut().pop_front();
        (self.update)();
        v
    }

    /// Retains only the elements specified by the predicate.
    pub fn retain<F>(&self, f: F)
    where
        T: Eq + Hash,
        F: FnMut(&T) -> bool,
    {
        self.inner.borrow_mut().retain(f);
        (self.update)();
    }

    /// Clears the queue, removing all values.
    pub fn clear(&self) {
        self.inner.borrow_mut().clear();
        (self.update)();
    }

    pub fn len(&self) -> usize {
        self.inner.borrow().len()
    }
}

impl<T> Clone for UseQueueHandle<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            update: self.update.clone(),
        }
    }
}

impl<T> PartialEq for UseQueueHandle<T>
where
    T: Eq + Hash,
{
    fn eq(&self, other: &Self) -> bool {
        *self.inner == *other.inner
    }
}

// A hook that tracks a queue and provides methods to modify it.
#[hook]
pub fn use_queue<T>(initial_value: VecDeque<T>) -> UseQueueHandle<T>
where
    T: 'static,
{
    let inner = use_mut_ref(|| initial_value);
    let update = use_update();

    UseQueueHandle { inner, update }
}
