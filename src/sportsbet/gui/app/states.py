"""State classes."""

import asyncio
from collections.abc import AsyncGenerator, Generator
from pathlib import Path
from typing import Any, Self, cast

import cloudpickle
import nest_asyncio
import numpy as np
import pandas as pd
import reflex as rx
from more_itertools import chunked
from reflex.event import EventSpec
from reflex_ag_grid import ag_grid
from sklearn.compose import make_column_transformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from sportsbet.datasets import BaseDataLoader, SoccerDataLoader
from sportsbet.evaluation import BaseBettor, BettorGridSearchCV, ClassifierBettor, OddsComparisonBettor, backtest

BETTING_MARKETS = [
    [
        'home_win__full_time_goals',
        'away_win__full_time_goals',
        'draw__full_time_goals',
        'over_2.5__full_time_goals',
        'under_2.5__full_time_goals',
    ],
    ['draw__full_time_goals', 'over_2.5__full_time_goals'],
    ['home_win__full_time_goals', 'away_win__full_time_goals'],
]
DATALOADERS = {
    '足球': SoccerDataLoader,
}
MODELS = {
    '赔率比较': BettorGridSearchCV(
        estimator=OddsComparisonBettor(),
        param_grid={
            'alpha': np.linspace(0.0, 0.05, 20),
            'betting_markets': BETTING_MARKETS,
        },
        error_score='raise',
    ),
    '逻辑回归': BettorGridSearchCV(
        estimator=ClassifierBettor(
            classifier=make_pipeline(
                make_column_transformer(
                    (OneHotEncoder(handle_unknown='ignore'), ['league', 'home_team', 'away_team']),
                    remainder='passthrough',
                    force_int_remainder_cols=False,
                ),
                SimpleImputer(),
                MultiOutputClassifier(
                    LogisticRegression(solver='liblinear', random_state=7, max_iter=int(1e5)),
                ),
            ),
        ),
        param_grid={
            'betting_markets': BETTING_MARKETS,
            'classifier__multioutputclassifier__estimator__C': [0.1, 1.0, 50.0],
        },
        error_score='raise',
    ),
    '梯度提升': BettorGridSearchCV(
        estimator=ClassifierBettor(
            classifier=make_pipeline(
                make_column_transformer(
                    (OneHotEncoder(handle_unknown='ignore'), ['league', 'home_team', 'away_team']),
                    remainder='passthrough',
                    sparse_threshold=0,
                    force_int_remainder_cols=False,
                ),
                SimpleImputer(),
                MultiOutputClassifier(HistGradientBoostingClassifier(random_state=10)),
            ),
        ),
        param_grid={
            'betting_markets': BETTING_MARKETS,
            'classifier__multioutputclassifier__estimator__max_depth': [3, 5, 8],
        },
        error_score='raise',
    ),
}
DEFAULT_PARAM_CHECKED = {
    'leagues': [
        '"England"',
        '"Spain"',
        '"France"',
    ],
    'years': [
        '2020',
        '2021',
        '2022',
        '2023',
        '2024',
        '2025',
    ],
    'divisions': ['1'],
}
VISIBILITY_LEVELS_DATALOADER_CREATION = {
    'sport': 2,
    'parameters': 3,
    'training_parameters': 4,
    'control': 5,
}
VISIBILITY_LEVELS_DATALOADER_LOADING = {
    'dataloader': 2,
    'control': 3,
}
VISIBILITY_LEVELS_MODEL_CREATION = {
    'model': 2,
    'dataloader': 3,
    'evaluation': 4,
    'control': 5,
}
VISIBILITY_LEVELS_MODEL_LOADING = {
    'dataloader_model': 2,
    'evaluation': 3,
    'control': 4,
}
DELAY = 0.001


nest_asyncio.apply()


class State(rx.State):
    """The index page state."""

    dataloader_error: bool = False
    model_error: bool = False

    # Elements
    visibility_level: int = 1
    loading: bool = False

    # Mode
    mode_category: str = '数据'
    mode_type: str = '新建'

    # Message
    streamed_message: str = ''

    @staticmethod
    def process_cols(col: str) -> str:
        """Proces a column."""
        return " ".join([" ".join(token.split('_')).title() for token in col.split('__')])

    async def on_load(self: Self) -> AsyncGenerator:
        """Event on page load."""
        message = """你可以创建或加载数据加载器，用于获取历史数据和赛程数据。
        你也可以创建或加载投注模型，评估模型表现并为即将开始的比赛识别价值投注。
        <br><br>
        <strong>数据，新建</strong><br>
        创建新的数据加载器<br><br>

        <strong>数据，加载</strong><br>
        加载已有数据加载器。<br><br>

        <strong>建模，新建</strong><br>
        创建新的投注模型。<br><br>

        <strong>建模，加载</strong><br>
        加载已有投注模型。"""
        self.streamed_message = ''
        for char in message:
            await asyncio.sleep(DELAY)
            self.streamed_message += char
            yield

    @rx.event
    async def submit_state(self: Self) -> AsyncGenerator:
        """Submit handler."""
        self.loading = True
        yield
        if self.visibility_level == 1:
            self.loading = False
            yield
        self.visibility_level += 1

    @rx.event
    def reset_state(self: Self) -> None:
        """Reset handler."""

        self.dataloader_error = False
        self.model_error = False

        # Elements visibility
        self.visibility_level = 1
        self.loading: bool = False

        # Mode
        self.mode_category = '数据'
        self.mode_type = '新建'

        # Message
        self.streamed_message = ''


class DataloaderState(State):
    """The dataloader state."""

    all_leagues: list[list[str]] = []  # noqa: RUF012
    all_years: list[list[str]] = []  # noqa: RUF012
    all_divisions: list[list[str]] = []  # noqa: RUF012
    param_checked: dict[str | int, bool] = {}  # noqa: RUF012
    dataloader_filename: str | None = None
    data_title: str | None = None
    loading_db: bool = False


class DataloaderCreationState(DataloaderState):
    """The dataloader creation state."""

    sport_selection: str = '足球'
    all_params: list[dict[str, Any]] = []  # noqa: RUF012
    leagues: list[str] = []  # noqa: RUF012
    years: list[int] = []  # noqa: RUF012
    divisions: list[int] = []  # noqa: RUF012
    params: list[dict[str, Any]] = []  # noqa: RUF012
    default_param_checked: dict[str, list[str]] = DEFAULT_PARAM_CHECKED
    odds_types: list[str] = []  # noqa: RUF012
    param_grid: list[dict] = []  # noqa: RUF012
    odds_type: str = 'market_average'
    drop_na_thres: list = [0.0]  # noqa: RUF012
    dataloader_serialized: str | None = None
    X_train: list | None = None
    Y_train: list | None = None
    O_train: list | None = None
    X_train_cols: list | None = None
    Y_train_cols: list | None = None
    O_train_cols: list | None = None
    X_fix: list | None = None
    Y_fix: list | None = None
    O_fix: list | None = None
    X_fix_cols: list | None = None
    Y_fix_cols: list | None = None
    O_fix_cols: list | None = None
    data: list | None = None
    data_cols: list | None = None

    async def on_load(self: Self) -> AsyncGenerator:
        """Event on page load."""
        message = """请先选择运动项目。当前仅支持足球，后续会逐步增加更多项目。"""
        self.streamed_message = ''
        for char in message:
            await asyncio.sleep(DELAY)
            self.streamed_message += char
            yield

    @staticmethod
    def process_form_data(form_data: dict[str, str]) -> list[str]:
        """Process the form data."""
        return [key.replace('"', '') for key in form_data]

    @rx.event
    def download_dataloader(self: Self) -> EventSpec:
        """Download the dataloader."""
        dataloader = bytes(cast(str, self.dataloader_serialized), 'iso8859_16')
        return rx.download(data=dataloader, filename=self.dataloader_filename)

    @rx.event
    def switch_displayed_data_category(self: Self) -> Generator:
        """Switch the displayed data category."""
        self.loading_db = True
        yield
        if self.data in (self.X_train, self.Y_train, self.O_train):
            self.data = self.X_fix
            self.data_cols = self.X_fix_cols
            self.data_title = '赛程输入数据'
            self.loading_db = False
            yield
        elif self.data in (self.X_fix, self.O_fix):
            self.data = self.X_train
            self.data_cols = self.X_train_cols
            self.data_title = '训练输入数据'
            self.loading_db = False
            yield

    @rx.event
    def switch_displayed_data_type(self: Self) -> Generator:
        """Switch the displayed data type."""
        self.loading_db = True
        yield
        if self.data == self.X_train:
            self.data = self.Y_train
            self.data_cols = self.Y_train_cols
            self.data_title = '训练输出数据'
            self.loading_db = False
            yield
        elif self.data == self.Y_train:
            self.data = self.O_train
            self.data_cols = self.O_train_cols
            self.data_title = '训练赔率数据'
            self.loading_db = False
            yield
        elif self.data == self.O_train:
            self.data = self.X_train
            self.data_cols = self.X_train_cols
            self.data_title = '训练输入数据'
            self.loading_db = False
            yield
        elif self.data == self.X_fix:
            self.data = self.O_fix
            self.data_cols = self.O_fix_cols
            self.data_title = '赛程赔率数据'
            self.loading_db = False
            yield
        elif self.data == self.O_fix:
            self.data = self.X_fix
            self.data_cols = self.X_fix_cols
            self.data_title = '赛程输入数据'
            self.loading_db = False
            yield

    @rx.event
    def update_param_checked(self: Self, name: str | int, checked: bool) -> None:
        """Update the parameters."""
        if isinstance(name, str):
            name = f'"{name}"'
        self.param_checked[name] = checked

    def update_params(self: Self) -> None:
        """Update the parameters grid."""
        self.params = [
            params
            for params in self.all_params
            if params['league'] in self.leagues
            and params['year'] in self.years
            and params['division'] in self.divisions
        ]

    @rx.event
    def handle_submit_leagues(self: Self, leagues_form_data: dict) -> None:
        """Handle the form submit."""
        self.leagues = self.process_form_data(leagues_form_data)
        self.update_params()

    @rx.event
    def handle_submit_years(self: Self, years_form_data: dict) -> None:
        """Handle the form submit."""
        self.years = [int(year) for year in self.process_form_data(years_form_data)]
        self.update_params()

    @rx.event
    def handle_submit_divisions(self: Self, divisions_form_data: dict) -> None:
        """Handle the form submit."""
        self.divisions = [int(division) for division in self.process_form_data(divisions_form_data)]
        self.update_params()

    @rx.event
    async def submit_state(self: Self) -> AsyncGenerator:
        """Submit handler."""
        self.loading = True
        yield
        if self.visibility_level == VISIBILITY_LEVELS_DATALOADER_CREATION['sport']:
            self.dataloader_filename = 'dataloader.pkl'
            self.all_params = DATALOADERS[self.sport_selection].get_all_params()
            self.all_leagues = list(chunked(sorted({params['league'] for params in self.all_params}), 6))
            self.all_years = list(chunked(sorted({params['year'] for params in self.all_params}), 5))
            self.all_divisions = list(chunked(sorted({params['division'] for params in self.all_params}), 1))
            self.leagues = [league.replace('"', '') for league in DEFAULT_PARAM_CHECKED['leagues']]
            self.years = [int(year) for year in DEFAULT_PARAM_CHECKED['years']]
            self.divisions = [int(division) for division in DEFAULT_PARAM_CHECKED['divisions']]
            self.loading = False
            yield
            message = """你可以通过选择训练数据范围来配置数据加载器。
            赛程数据会遵循与训练数据一致的字段结构，从而保证训练与推理阶段的一致性。<br><br>

            <strong>训练数据</strong><br>
            选择要纳入的联赛、级别和年份。<br><br>

            <strong>赛程数据</strong><br>
            联赛、级别和年份的选择不会影响赛程数据，赛程包含所有即将进行的比赛。"""
            self.streamed_message = ''
            for char in message:
                await asyncio.sleep(DELAY)
                self.streamed_message += char
                yield
        elif self.visibility_level == VISIBILITY_LEVELS_DATALOADER_CREATION['parameters']:
            self.update_params()
            self.param_grid = [{k: [v] for k, v in param.items()} for param in self.params]
            self.odds_types = DATALOADERS[self.sport_selection](self.param_grid).get_odds_types()
            self.loading = False
            yield
            message = """训练数据包含输入、输出和赔率；赛程数据仅包含输入和赔率。<br><br>

            <strong>训练数据</strong><br>
            你可以选择要使用的赔率类型。
            同时可以设置训练数据的缺失值容忍阈值，超过阈值的列将被移除。<br><br>

            <strong>赛程数据</strong><br>
            由于赛程数据与训练数据字段结构保持一致，训练阶段的选择会影响赛程数据。"""
            self.streamed_message = ''
            for char in message:
                await asyncio.sleep(DELAY)
                self.streamed_message += char
                yield
        elif self.visibility_level == VISIBILITY_LEVELS_DATALOADER_CREATION['training_parameters']:
            dataloader = DATALOADERS[self.sport_selection](self.param_grid)
            X_train, Y_train, O_train = dataloader.extract_train_data(
                odds_type=self.odds_type,
                drop_na_thres=float(self.drop_na_thres[0]),
            )
            X_fix, _, O_fix = dataloader.extract_fixtures_data()
            self.data = self.X_train = X_train.reset_index().fillna('NA').to_dict('records')
            self.data_cols = self.X_train_cols = [ag_grid.column_def(field='date', header_name='Date')] + [
                ag_grid.column_def(field=col, header_name=self.process_cols(col)) for col in X_train.columns
            ]
            self.data_title = '训练输入数据'
            self.Y_train = Y_train.fillna('NA').to_dict('records')
            self.Y_train_cols = [
                ag_grid.column_def(field=col, header_name=self.process_cols(col)) for col in Y_train.columns
            ]
            self.O_train = O_train.fillna('NA').to_dict('records') if O_train is not None else None
            self.O_train_cols = (
                [ag_grid.column_def(field=col, header_name=self.process_cols(col)) for col in O_train.columns]
                if O_train is not None
                else None
            )
            self.X_fix = X_fix.reset_index().fillna('NA').to_dict('records')
            self.X_fix_cols = [ag_grid.column_def(field='date', header_name='Date')] + [
                ag_grid.column_def(field=col, header_name=self.process_cols(col)) for col in X_fix.columns
            ]
            self.O_fix = O_fix.fillna('NA').to_dict('records') if O_fix is not None else None
            self.O_fix_cols = (
                [ag_grid.column_def(field=col, header_name=self.process_cols(col)) for col in O_fix.columns]
                if O_fix is not None
                else None
            )
            self.dataloader_serialized = str(cloudpickle.dumps(dataloader), 'iso8859_16')
            self.loading = False
            yield
        self.visibility_level += 1

    @rx.event
    def reset_state(self: Self) -> None:
        """Reset handler."""

        self.dataloader_error = False
        self.model_error = False

        # Elements visibility
        self.visibility_level = 1
        self.loading: bool = False

        # Mode
        self.mode_category = '数据'
        self.mode_type = '新建'

        # Data
        self.dataloader_serialized = None
        self.dataloader_filename = None
        self.all_leagues = []
        self.all_years = []
        self.all_divisions = []
        self.param_checked = {}
        self.odds_type = None
        self.drop_na_thres = None

        # Data
        self.data = None
        self.data_cols = None
        self.data_title = None
        self.loading_db = False
        self.X_train = None
        self.Y_train = None
        self.O_train = None
        self.X_train_cols = None
        self.Y_train_cols = None
        self.O_train_cols = None
        self.X_fix = None
        self.O_fix = None
        self.X_fix_cols = None
        self.O_fix_cols = None

        # Message
        self.streamed_message = ''


class DataloaderLoadingState(DataloaderState):
    """The dataloader loading state."""

    odds_type: str | None = None
    drop_na_thres: float | None = None
    dataloader_serialized: str | None = None
    X_train: list | None = None
    Y_train: list | None = None
    O_train: list | None = None
    X_train_cols: list | None = None
    Y_train_cols: list | None = None
    O_train_cols: list | None = None
    X_fix: list | None = None
    Y_fix: list | None = None
    O_fix: list | None = None
    X_fix_cols: list | None = None
    Y_fix_cols: list | None = None
    O_fix_cols: list | None = None
    data: list | None = None
    data_cols: list | None = None

    async def on_load(self: Self) -> AsyncGenerator:
        """Event on page load."""
        message = """请选择数据加载器文件，以提取最新的训练数据和赛程数据。"""
        self.streamed_message = ''
        for char in message:
            await asyncio.sleep(DELAY)
            self.streamed_message += char
            yield

    @rx.event
    async def handle_dataloader_upload(self: Self, files: list[rx.UploadFile]) -> AsyncGenerator:
        """Handle the upload of files."""
        self.loading = True
        yield
        for file in files:
            dataloader = await file.read()
            self.dataloader_serialized = str(dataloader, 'iso8859_16')
            self.dataloader_filename = Path(file.filename).name
        self.loading = False
        yield
        dataloader = cloudpickle.loads(bytes(cast(str, self.dataloader_serialized), 'iso8859_16'))
        if not isinstance(dataloader, BaseDataLoader):
            self.dataloader_error = True
            message = """上传文件不是数据加载器，请重试。"""
            self.streamed_message = ''
            for char in message:
                await asyncio.sleep(DELAY)
                self.streamed_message += char
                yield
        else:
            self.dataloader_error = False
            message = """上传文件是有效的数据加载器，可以继续下一步。"""
            self.streamed_message = ''
            for char in message:
                await asyncio.sleep(DELAY)
                self.streamed_message += char
                yield

    @rx.event
    def download_dataloader(self: Self) -> EventSpec:
        """Download the dataloader."""
        dataloader = bytes(cast(str, self.dataloader_serialized), 'iso8859_16')
        return rx.download(data=dataloader, filename=self.dataloader_filename)

    @rx.event
    def switch_displayed_data_category(self: Self) -> Generator:
        """Switch the displayed data category."""
        self.loading_db = True
        yield
        if self.data in (self.X_train, self.Y_train, self.O_train):
            self.data = self.X_fix
            self.data_cols = self.X_fix_cols
            self.data_title = '赛程输入数据'
            self.loading_db = False
            yield
        elif self.data in (self.X_fix, self.O_fix):
            self.data = self.X_train
            self.data_cols = self.X_train_cols
            self.data_title = '训练输入数据'
            self.loading_db = False
            yield

    @rx.event
    def switch_displayed_data_type(self: Self) -> Generator:
        """Switch the displayed data type."""
        self.loading_db = True
        yield
        if self.data == self.X_train:
            self.data = self.Y_train
            self.data_cols = self.Y_train_cols
            self.data_title = '训练输出数据'
            self.loading_db = False
            yield
        elif self.data == self.Y_train:
            self.data = self.O_train
            self.data_cols = self.O_train_cols
            self.data_title = '训练赔率数据'
            self.loading_db = False
            yield
        elif self.data == self.O_train:
            self.data = self.X_train
            self.data_cols = self.X_train_cols
            self.data_title = '训练输入数据'
            self.loading_db = False
            yield
        elif self.data == self.X_fix:
            self.data = self.O_fix
            self.data_cols = self.O_fix_cols
            self.data_title = '赛程赔率数据'
            self.loading_db = False
            yield
        elif self.data == self.O_fix:
            self.data = self.X_fix
            self.data_cols = self.X_fix_cols
            self.data_title = '赛程输入数据'
            self.loading_db = False
            yield

    @rx.event
    async def submit_state(self: Self) -> AsyncGenerator:
        """Submit handler."""
        self.loading = True
        yield
        if self.visibility_level == VISIBILITY_LEVELS_DATALOADER_LOADING['dataloader']:
            dataloader = cloudpickle.loads(bytes(cast(str, self.dataloader_serialized), 'iso8859_16'))
            if hasattr(dataloader, 'odds_type_') and hasattr(dataloader, 'drop_na_thres_'):
                X_train, Y_train, O_train = dataloader.extract_train_data(
                    odds_type=dataloader.odds_type_,
                    drop_na_thres=dataloader.drop_na_thres_,
                )
            else:
                X_train, Y_train, O_train = dataloader.extract_train_data()
            X_fix, _, O_fix = dataloader.extract_fixtures_data()
            self.data = self.X_train = X_train.reset_index().fillna('NA').to_dict('records')
            self.data_cols = self.X_train_cols = [ag_grid.column_def(field='date', header_name='Date')] + [
                ag_grid.column_def(field=col, header_name=self.process_cols(col)) for col in X_train.columns
            ]
            self.data_title = '训练输入数据'
            self.Y_train = Y_train.fillna('NA').to_dict('records')
            self.Y_train_cols = [
                ag_grid.column_def(field=col, header_name=self.process_cols(col)) for col in Y_train.columns
            ]
            self.O_train = O_train.fillna('NA').to_dict('records') if O_train is not None else None
            self.O_train_cols = (
                [ag_grid.column_def(field=col, header_name=self.process_cols(col)) for col in O_train.columns]
                if O_train is not None
                else None
            )
            self.X_fix = X_fix.reset_index().fillna('NA').to_dict('records')
            self.X_fix_cols = [ag_grid.column_def(field='date', header_name='Date')] + [
                ag_grid.column_def(field=col, header_name=self.process_cols(col)) for col in X_fix.columns
            ]
            self.O_fix = O_fix.fillna('NA').to_dict('records') if O_fix is not None else None
            self.O_fix_cols = (
                [ag_grid.column_def(field=col, header_name=self.process_cols(col)) for col in O_fix.columns]
                if O_fix is not None
                else None
            )
            all_params = dataloader.get_all_params()
            self.all_leagues = list(chunked(sorted({params['league'] for params in all_params}), 6))
            self.all_years = list(chunked(sorted({params['year'] for params in all_params}), 5))
            self.all_divisions = list(chunked(sorted({params['division'] for params in all_params}), 1))
            self.param_checked = {
                **{f'"{key}"': True for key in {params['league'] for params in dataloader.param_grid_}},
                **dict.fromkeys({params['year'] for params in dataloader.param_grid_}, True),
                **dict.fromkeys({params['division'] for params in dataloader.param_grid_}, True),
            }
            self.odds_type = dataloader.odds_type_
            self.drop_na_thres = dataloader.drop_na_thres_
            self.loading = False
            yield
        self.visibility_level += 1

    @rx.event
    def reset_state(self: Self) -> None:
        """Reset handler."""

        self.dataloader_error = False
        self.model_error = False

        # Elements visibility
        self.visibility_level = 1
        self.loading: bool = False

        # Mode
        self.mode_category = '数据'
        self.mode_type = '新建'

        # Data
        self.dataloader_serialized = None
        self.dataloader_filename = None
        self.all_leagues = []
        self.all_years = []
        self.all_divisions = []
        self.param_checked = {}
        self.odds_type = None
        self.drop_na_thres = None

        # Data
        self.data = None
        self.data_cols = None
        self.data_title = None
        self.loading_db = False
        self.X_train = None
        self.Y_train = None
        self.O_train = None
        self.X_train_cols = None
        self.Y_train_cols = None
        self.O_train_cols = None
        self.X_fix = None
        self.O_fix = None
        self.X_fix_cols = None
        self.O_fix_cols = None

        # Message
        self.streamed_message = ''


class ModelCreationState(State):
    """The model creation state."""

    model_selection: str = '赔率比较'
    dataloader_serialized: str | None = None
    dataloader_filename: str | None = None
    model_serialized: str | None = None
    model_filename: str | None = None
    evaluation_selection: str = '回测'
    backtesting_results: list | None = None
    backtesting_results_cols: list | None = None
    optimal_params: list | None = None
    optimal_params_cols: list | None = None
    value_bets: list | None = None
    value_bets_cols: list | None = None

    async def on_load(self: Self) -> AsyncGenerator:
        """Event on page load."""
        message = """请先选择投注模型。当前提供三种模型。<br><br>

        <strong>赔率比较模型</strong><br>
        基于平均赔率估计概率，并识别价值投注。<br><br>

        <strong>逻辑回归模型</strong><br>
        在训练数据上拟合逻辑回归分类器，支持多组超参数，并处理类别特征和缺失值。<br><br>

        <strong>梯度提升模型</strong><br>
        在训练数据上拟合梯度提升分类器，支持多组超参数，同样可处理类别特征和缺失值。"""
        self.streamed_message = ''
        for char in message:
            await asyncio.sleep(DELAY)
            self.streamed_message += char
            yield

    @rx.event
    def download_model(self: Self) -> EventSpec:
        """Download the model."""
        model = bytes(cast(str, self.model_serialized), 'iso8859_16')
        return rx.download(data=model, filename=self.model_filename)

    @rx.event
    async def handle_dataloader_upload(self: Self, files: list[rx.UploadFile]) -> AsyncGenerator:
        """Handle the upload of files."""
        self.loading = True
        yield
        for file in files:
            dataloader = await file.read()
            self.dataloader_serialized = str(dataloader, 'iso8859_16')
            self.dataloader_filename = Path(file.filename).name
        self.loading = False
        yield
        dataloader = cloudpickle.loads(bytes(cast(str, self.dataloader_serialized), 'iso8859_16'))
        if not isinstance(dataloader, BaseDataLoader):
            self.dataloader_error = True
            message = """上传文件不是数据加载器，请重试。"""
            self.streamed_message = ''
            for char in message:
                await asyncio.sleep(DELAY)
                self.streamed_message += char
                yield
        else:
            self.dataloader_error = False
            message = """上传文件是有效的数据加载器，可以继续下一步。"""
            self.streamed_message = ''
            for char in message:
                await asyncio.sleep(DELAY)
                self.streamed_message += char
                yield

    @rx.event
    async def submit_state(self: Self) -> AsyncGenerator:
        """Submit handler."""
        self.loading = True
        yield
        if self.visibility_level == VISIBILITY_LEVELS_MODEL_CREATION['model']:
            self.loading = False
            yield
            message = (
                """请上传数据加载器，以便对模型执行回测或识别价值投注。"""
            )
            self.streamed_message = ''
            for char in message:
                await asyncio.sleep(DELAY)
                self.streamed_message += char
                yield
        elif self.visibility_level == VISIBILITY_LEVELS_MODEL_CREATION['dataloader']:
            self.loading = False
            yield
            message = """请选择“回测”或“价值投注”运行方式。<br><br>

            回测使用 3 折时间序列交叉验证，单注固定为 50，初始资金为 10000。
            回测结束后，模型会在全部训练集上重新拟合。<br><br>

            价值投注模式会使用赛程数据进行预测。
            预测前模型同样会在全部训练集上完成拟合。"""
            self.streamed_message = ''
            for char in message:
                await asyncio.sleep(DELAY)
                self.streamed_message += char
                yield
        elif self.visibility_level == VISIBILITY_LEVELS_MODEL_CREATION['evaluation']:
            dataloader = cloudpickle.loads(bytes(cast(str, self.dataloader_serialized), 'iso8859_16'))
            if hasattr(dataloader, 'odds_type_') and hasattr(dataloader, 'drop_na_thres_'):
                X_train, Y_train, O_train = dataloader.extract_train_data(
                    odds_type=dataloader.odds_type_,
                    drop_na_thres=dataloader.drop_na_thres_,
                )
            else:
                X_train, Y_train, O_train = dataloader.extract_train_data()
            model = MODELS[self.model_selection]
            model.fit(X_train, Y_train, O_train)
            self.model_serialized = str(cloudpickle.dumps(model), 'iso8859_16')
            self.model_filename = 'model.pkl'
            if self.evaluation_selection == '回测':
                backtesting_results = backtest(model, X_train, Y_train, O_train, cv=TimeSeriesSplit(3)).reset_index()
                self.backtesting_results = backtesting_results.fillna('NA').to_dict('records')
                self.backtesting_results_cols = [
                    ag_grid.column_def(field=col, header_name=self.process_cols(col))
                    for col in backtesting_results.columns
                ]
                self.optimal_params = [
                    {'参数名': name, '最优值': value} for name, value in model.best_params_.items()
                ]
                self.optimal_params_cols = [
                    ag_grid.column_def(field='参数名'),
                    ag_grid.column_def(field='最优值'),
                ]
            elif self.evaluation_selection == '价值投注':
                X_fix, *_ = dataloader.extract_fixtures_data()
                value_bets = pd.DataFrame(np.round(1 / model.predict_proba(X_fix), 2), columns=model.betting_markets_)
                value_bets = pd.concat(
                    [X_fix.reset_index()[['date', 'league', 'division', 'home_team', 'away_team']], value_bets],
                    axis=1,
                )
                self.value_bets = value_bets.fillna('NA').to_dict('records')
                self.value_bets_cols = [
                    ag_grid.column_def(field=col, header_name=self.process_cols(col)) for col in value_bets.columns
                ]
            self.loading = False
            yield
        self.visibility_level += 1

    @rx.event
    def reset_state(self: Self) -> None:
        """Reset handler."""

        self.dataloader_error = False
        self.model_error = False

        # Elements visibility
        self.visibility_level = 1
        self.loading: bool = False

        # Mode
        self.mode_category = '数据'
        self.mode_type = '新建'

        # Model
        self.model_selection = '赔率比较'

        # Data
        self.dataloader_serialized = None
        self.dataloader_filename = None

        # Evaluation
        self.model_serialized = None
        self.model_filename = None
        self.evaluation_selection = '回测'
        self.backtesting_results = None
        self.backtesting_results_cols = None
        self.optimal_params = None
        self.optimal_params_cols = None
        self.value_bets = None
        self.value_bets_cols = None

        # Message
        self.streamed_message = ''


class ModelLoadingState(State):
    """The model loading state."""

    dataloader_serialized: str | None = None
    dataloader_filename: str | None = None
    model_serialized: str | None = None
    model_filename: str | None = None
    evaluation_selection: str = '回测'
    backtesting_results: list | None = None
    backtesting_results_cols: list | None = None
    optimal_params: list | None = None
    optimal_params_cols: list | None = None
    value_bets: list | None = None
    value_bets_cols: list | None = None

    async def on_load(self: Self) -> AsyncGenerator:
        """Event on page load."""
        message = """请上传数据加载器和投注模型，以执行回测或识别价值投注。"""
        self.streamed_message = ''
        for char in message:
            await asyncio.sleep(DELAY)
            self.streamed_message += char
            yield

    @rx.event
    def download_model(self: Self) -> EventSpec:
        """Download the model."""
        model = bytes(cast(str, self.model_serialized), 'iso8859_16')
        return rx.download(data=model, filename=self.model_filename)

    @rx.event
    async def handle_dataloader_upload(self: Self, files: list[rx.UploadFile]) -> AsyncGenerator:
        """Handle the upload of dataloader files."""
        self.loading = True
        yield
        for file in files:
            dataloader = await file.read()
            self.dataloader_serialized = str(dataloader, 'iso8859_16')
            self.dataloader_filename = Path(file.filename).name
        self.loading = False
        yield
        dataloader = cloudpickle.loads(bytes(cast(str, self.dataloader_serialized), 'iso8859_16'))
        if not isinstance(dataloader, BaseDataLoader):
            self.dataloader_error = True
            message = """上传文件不是数据加载器，请重试。"""
            self.streamed_message = ''
            for char in message:
                await asyncio.sleep(DELAY)
                self.streamed_message += char
                yield
        else:
            self.dataloader_error = False
            message = """上传文件是有效的数据加载器，可以继续下一步。"""
            self.streamed_message = ''
            for char in message:
                await asyncio.sleep(DELAY)
                self.streamed_message += char
                yield

    @rx.event
    async def handle_model_upload(self: Self, files: list[rx.UploadFile]) -> AsyncGenerator:
        """Handle the upload of model files."""
        self.loading = True
        yield
        for file in files:
            model = await file.read()
            self.model_serialized = str(model, 'iso8859_16')
            self.model_filename = Path(file.filename).name
        self.loading = False
        yield
        model = cloudpickle.loads(bytes(cast(str, self.model_serialized), 'iso8859_16'))
        if not isinstance(model, BaseBettor):
            self.model_error = True
            message = """上传文件不是投注模型，请重试。"""
            self.streamed_message = ''
            for char in message:
                await asyncio.sleep(DELAY)
                self.streamed_message += char
                yield
        else:
            self.model_error = False
            message = """上传文件是有效的投注模型，可以继续下一步。"""
            self.streamed_message = ''
            for char in message:
                await asyncio.sleep(DELAY)
                self.streamed_message += char
                yield

    @rx.event
    async def submit_state(self: Self) -> AsyncGenerator:
        """Submit handler."""
        self.loading = True
        yield
        if self.visibility_level == VISIBILITY_LEVELS_MODEL_LOADING['dataloader_model']:
            self.loading = False
            yield
            message = """请选择“回测”或“价值投注”运行方式。<br><br>

            回测使用 3 折时间序列交叉验证，单注固定为 50，初始资金为 10000。
            回测结束后，模型会在全部训练集上重新拟合。<br><br>

            价值投注模式会使用赛程数据进行预测。
            预测前模型同样会在全部训练集上完成拟合。"""
            self.streamed_message = ''
            for char in message:
                await asyncio.sleep(DELAY)
                self.streamed_message += char
                yield
        elif self.visibility_level == VISIBILITY_LEVELS_MODEL_LOADING['evaluation']:
            dataloader = cloudpickle.loads(bytes(cast(str, self.dataloader_serialized), 'iso8859_16'))
            if hasattr(dataloader, 'odds_type_') and hasattr(dataloader, 'drop_na_thres_'):
                X_train, Y_train, O_train = dataloader.extract_train_data(
                    odds_type=dataloader.odds_type_,
                    drop_na_thres=dataloader.drop_na_thres_,
                )
            else:
                X_train, Y_train, O_train = dataloader.extract_train_data()
            model = cloudpickle.loads(bytes(cast(str, self.model_serialized), 'iso8859_16'))
            model.fit(X_train, Y_train, O_train)
            self.model_serialized = str(cloudpickle.dumps(model), 'iso8859_16')
            self.model_filename = 'model.pkl'
            if self.evaluation_selection == '回测':
                backtesting_results = backtest(model, X_train, Y_train, O_train, cv=TimeSeriesSplit(3)).reset_index()
                self.backtesting_results = backtesting_results.fillna('NA').to_dict('records')
                self.backtesting_results_cols = [
                    ag_grid.column_def(field=col, header_name=self.process_cols(col))
                    for col in backtesting_results.columns
                ]
                self.optimal_params = [
                    {'参数名': name, '最优值': value} for name, value in model.best_params_.items()
                ]
                self.optimal_params_cols = [
                    ag_grid.column_def(field='参数名'),
                    ag_grid.column_def(field='最优值'),
                ]
            elif self.evaluation_selection == '价值投注':
                X_fix, *_ = dataloader.extract_fixtures_data()
                value_bets = pd.DataFrame(np.round(1 / model.predict_proba(X_fix), 2), columns=model.betting_markets_)
                value_bets = pd.concat(
                    [X_fix.reset_index()[['date', 'league', 'division', 'home_team', 'away_team']], value_bets],
                    axis=1,
                )
                self.value_bets = value_bets.fillna('NA').to_dict('records')
                self.value_bets_cols = [
                    ag_grid.column_def(field=col, header_name=self.process_cols(col)) for col in value_bets.columns
                ]
            self.loading = False
            yield
        self.visibility_level += 1

    @rx.event
    def reset_state(self: Self) -> None:
        """Reset handler."""

        self.dataloader_error = False
        self.model_error = False

        # Elements visibility
        self.visibility_level = 1
        self.loading: bool = False

        # Mode
        self.mode_category = '数据'
        self.mode_type = '新建'

        # Data
        self.dataloader_serialized = None
        self.dataloader_filename = None

        # Evaluation
        self.model_serialized = None
        self.model_filename = None
        self.evaluation_selection = '回测'
        self.backtesting_results = None
        self.backtesting_results_cols = None
        self.optimal_params = None
        self.optimal_params_cols = None
        self.value_bets = None
        self.value_bets_cols = None

        # Message
        self.streamed_message = ''
