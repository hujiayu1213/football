"""Index page."""

import reflex as rx

from .components import bot, control, mode, navbar, save_dataloader, save_model, sidebar
from .states import (
    DataloaderCreationState,
    DataloaderLoadingState,
    ModelCreationState,
    ModelLoadingState,
    State,
)


@rx.page(route="/", on_load=State.on_load)
def index() -> rx.Component:
    """Index page."""
    return rx.box(
        navbar(),
        rx.hstack(
            rx.vstack(
                rx.cond(
                    (State.mode_category == '数据') & (State.mode_type == '新建'),
                    sidebar(
                        mode(State, '创建数据加载器'),
                        control=control(State, False, save_dataloader(DataloaderCreationState), '/dataloader/creation'),
                    ),
                ),
                rx.cond(
                    (State.mode_category == '数据') & (State.mode_type == '加载'),
                    sidebar(
                        mode(State, '加载数据加载器'),
                        control=control(State, False, save_dataloader(DataloaderLoadingState), '/dataloader/loading'),
                    ),
                ),
                rx.cond(
                    (State.mode_category == '建模') & (State.mode_type == '新建'),
                    sidebar(
                        mode(State, '创建模型'),
                        control=control(State, False, save_model(ModelCreationState), '/model/creation'),
                    ),
                ),
                rx.cond(
                    (State.mode_category == '建模') & (State.mode_type == '加载'),
                    sidebar(
                        mode(State, '加载模型'),
                        control=control(State, False, save_model(ModelLoadingState), '/model/loading'),
                    ),
                ),
            ),
            bot(State, 2),
            padding='2%',
        ),
    )
