"""Dataloader creation page."""

from collections.abc import Callable
from typing import cast

import reflex as rx

from .components import bot, control, dataloader_data, mode, navbar, save_dataloader, sidebar, title
from .states import VISIBILITY_LEVELS_DATALOADER_CREATION as VL
from .states import DataloaderCreationState


def sport(state: rx.State) -> rx.Component:
    """The sport component."""
    return rx.vstack(
        title('运动项目', 'medal'),
        rx.text('请选择运动项目', size='1'),
        rx.select(
            items=['足球'],
            value='足球',
            disabled=state.visibility_level > VL['sport'],
            on_change=state.set_sport_selection,
        ),
    )


def parameters(state: rx.State) -> rx.Component:
    """The parameters component."""

    def _in_leagues(name: rx.Var) -> rx.Var:
        return state.default_param_checked['leagues'].contains(name.to_string())

    def _in_years(name: rx.Var) -> rx.Var:
        return state.default_param_checked['years'].contains(name.to_string())

    def _in_divisions(name: rx.Var) -> rx.Var:
        return state.default_param_checked['divisions'].contains(name.to_string())

    def _checkboxes(row: list[str], state: rx.State) -> rx.Component:
        """Checkbox of parameter value."""

        return rx.vstack(
            rx.foreach(
                row,
                lambda name: rx.checkbox(
                    name,
                    default_checked=rx.cond(
                        _in_leagues(name),
                        True,
                        rx.cond(_in_years(name), True, rx.cond(_in_divisions(name), True, False)),
                    ),
                    checked=state.param_checked[name.to_string()],
                    name=name.to_string(),
                    on_change=lambda checked: state.update_param_checked(name, checked),
                ),
            ),
        )

    def _dialog(name: str, icon_name: str, rows: list[list[str]], on_submit: Callable) -> rx.Component:
        """The dialog component."""
        return rx.dialog.root(
            rx.dialog.trigger(
                rx.button(
                    rx.tooltip(rx.icon(icon_name), content=name),
                    size='4',
                    variant='outline',
                    disabled=state.visibility_level > VL['parameters'],
                ),
            ),
            rx.dialog.content(
                rx.form.root(
                    rx.dialog.title(name),
                    rx.dialog.description(
                        f'选择需要包含在训练数据中的{name}。',
                        size="2",
                        margin_bottom="16px",
                    ),
                    rx.hstack(rx.foreach(rows, lambda row: _checkboxes(row, state))),
                    rx.flex(
                        rx.dialog.close(rx.button('提交', type='submit')),
                        justify='end',
                        spacing="3",
                        margin_top="50px",
                    ),
                    on_submit=on_submit,
                    reset_on_submit=False,
                    width="100%",
                ),
            ),
        )

    return rx.cond(
        state.visibility_level > VL['sport'],
        rx.vstack(
            title('参数', 'proportions'),
            rx.text('请选择参数', size='1'),
            rx.hstack(
                _dialog('联赛', 'earth', state.all_leagues, state.handle_submit_leagues),
                _dialog('年份', 'calendar', state.all_years, state.handle_submit_years),
                _dialog('级别', 'gauge', state.all_divisions, state.handle_submit_divisions),
            ),
            rx.cond(
                ~cast(rx.Var, DataloaderCreationState).leagues.bool(),
                rx.text('联赛为空，请至少选择一项。', size='1'),
            ),
            rx.cond(
                ~cast(rx.Var, DataloaderCreationState).years.bool(),
                rx.text('年份为空，请至少选择一项。', size='1'),
            ),
            rx.cond(
                ~cast(rx.Var, DataloaderCreationState).divisions.bool(),
                rx.text('级别为空，请至少选择一项。', size='1'),
            ),
        ),
    )


def training_parameters(state: rx.State) -> rx.Component:
    """The training parameters selection component."""
    return rx.cond(
        DataloaderCreationState.visibility_level > VL['parameters'],
        rx.vstack(
            rx.vstack(
                rx.text('赔率类型', size='1'),
                rx.select(
                    state.odds_types,
                    default_value=state.odds_types[0],
                    on_change=state.set_odds_type,
                    disabled=state.visibility_level > VL['training_parameters'],
                ),
            ),
            rx.vstack(
                rx.text(f'列缺失值剔除阈值: {DataloaderCreationState.drop_na_thres}', size='1'),
                rx.slider(
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    default_value=0.0,
                    on_change=state.set_drop_na_thres,
                    disabled=state.visibility_level > VL['training_parameters'],
                ),
            ),
        ),
    )


@rx.page(route="/dataloader/creation", on_load=DataloaderCreationState.on_load)
def dataloader_creation_page() -> rx.Component:
    """Main page."""
    return rx.box(
        navbar(),
        rx.hstack(
            sidebar(
                mode(DataloaderCreationState, '创建数据加载器'),
                sport(DataloaderCreationState),
                parameters(DataloaderCreationState),
                training_parameters(DataloaderCreationState),
                control=control(
                    DataloaderCreationState,
                    (DataloaderCreationState.visibility_level == VL['control'])
                    | (
                        (DataloaderCreationState.visibility_level == VL['parameters'])
                        & (
                            (~cast(rx.Var, DataloaderCreationState).leagues.bool())
                            | (~cast(rx.Var, DataloaderCreationState).years.bool())
                            | (~cast(rx.Var, DataloaderCreationState).divisions.bool())
                        )
                    ),
                    save=save_dataloader(DataloaderCreationState, VL['control']),
                ),
            ),
            dataloader_data(DataloaderCreationState, VL['control']),
            bot(DataloaderCreationState, VL['control']),
            padding='2%',
        ),
    )
