use yew::prelude::*;
use yew_router::prelude::*;

pub mod home;

use home::Home;

/// App routes
#[derive(Routable, Debug, Clone, PartialEq, Eq)]
pub enum AppRoute {
    #[not_found]
    #[at("/page-not-found")]
    PageNotFound,
    #[at("/")]
    Home,
}

/// Switch app routes
pub fn switch(routes: AppRoute) -> Html {
    match routes.clone() {
        AppRoute::Home => html! { <Home /> },
        AppRoute::PageNotFound => html! { "Page not found" },
    }
}

/// Root app component
#[function_component(App)]
pub fn app() -> Html {
    html! {
        <HashRouter>
            <div class="flex min-h-screen flex-col">
                <Switch<AppRoute> render={switch} />
            </div>
        </HashRouter>
    }
}
