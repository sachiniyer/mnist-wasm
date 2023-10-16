use crate::app::api::get_sample;
use yew::prelude::*;
use yew::Properties;

#[derive(Properties, Clone, PartialEq)]
pub struct GridCellProps {
    pub row: usize,
    pub col: usize,
    pub set_cell: Callback<(usize, usize)>,
    pub mouse_down: bool,
    pub val: bool,
}

#[function_component(GridCell)]
pub fn grid_cell(props: &GridCellProps) -> Html {
    let change = {
        if !props.mouse_down {
            Callback::noop()
        } else {
            let props = props.clone();
            Callback::from(move |_| {
                props.set_cell.emit((props.row.clone(), props.col.clone()));
            })
        }
    };
    fn color(val: bool) -> &'static str {
        if val {
            "background-color: black"
        } else {
            "background-color: white"
        }
    }
    html! {
        <div style={ color(props.val) }
            class="w-4 h-4 border border-gray-400"
            onmouseover={ change }
        >
        </div>
    }
}

#[derive(Properties, Clone, PartialEq)]
pub struct GridProps {
    pub grid: Callback<[[bool; 28]; 28]>,
    pub init_grid: [[bool; 28]; 28],
}

#[function_component(Grid)]
pub fn grid(props: &GridProps) -> Html {
    let mouse_down_handle = use_state(|| false);
    let mouse_down = (*mouse_down_handle).clone();

    let grid_local_handler = use_state(|| props.init_grid.clone());

    let grid_local_modify = {
        let grid_local_handler = grid_local_handler.clone();
        let props = props.clone();
        Callback::from(move |(row, col): (usize, usize)| {
            let mut grid_local = (*grid_local_handler).clone();
            grid_local[row][col] = true;
            props.grid.emit(grid_local);
            grid_local_handler.set(grid_local);
        })
    };

    let clear_grid = {
        let grid_local_handler = grid_local_handler.clone();
        let props = props.clone();
        Callback::from(move |_| grid_local_handler.set(props.init_grid.clone()))
    };

    let load_sample = {
        let grid_local_handler = grid_local_handler.clone();
        let props = props.clone();
        Callback::from(move |_| {
            let grid_local_handler = grid_local_handler.clone();
            let props = props.clone();
            wasm_bindgen_futures::spawn_local(async move {
                let sample = get_sample().await;
                let vec_image = sample
                    .image
                    .iter()
                    .map(|&x| x > 0.0)
                    .collect::<Vec<bool>>()
                    .chunks(28)
                    .map(|x| x.to_vec())
                    .collect::<Vec<Vec<bool>>>();
                let mut grid = [[false; 28]; 28];
                for (i, row) in vec_image.iter().enumerate() {
                    for (j, col) in row.iter().enumerate() {
                        grid[i][j] = *col;
                    }
                }
                props.grid.emit(grid);
                grid_local_handler.set(grid)
            });
        })
    };

    let mut grid_display = vec![];
    for row in 0..28 {
        let mut row_html = vec![];

        for col in 0..28 {
            row_html.push(html! {
                <GridCell row={row}
                          col={col}
                          set_cell={grid_local_modify.clone()}
                          val={grid_local_handler[row][col]}
                          mouse_down={mouse_down} />
            });
        }

        grid_display.push(html! {
                <div class="flex flex-row">
                    { row_html }
                </div>
        });
    }

    html! {
        <>
        <div
            onmousedown={
                let mouse_down_handle = mouse_down_handle.clone();
                Callback::from(move |_| mouse_down_handle.set(true))
            }
            onmouseup={
                let mouse_down_handle = mouse_down_handle.clone();
                Callback::from(move |_| mouse_down_handle.set(false))
            }
            class="flex flex-col">
            { grid_display }
        </div>
        <button class="grid-control" onclick={clear_grid}>{ "Clear Grid" }</button>
        <button class="grid-control" onclick={load_sample}>{ "Load Sample" }</button>
        </>
    }
}
