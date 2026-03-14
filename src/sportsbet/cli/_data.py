"""Module that contains the datasets functionality of the CLI."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from pathlib import Path

import click
import pandas as pd
from rich.console import Console
from rich.panel import Panel

from ._options import get_config_path_option, get_data_path_option
from ._utils import get_dataloader_cls, get_drop_na_thres, get_module, get_odds_type, get_param_grid, print_console


@click.group()
def dataloader() -> None:
    """使用或创建数据加载器。"""
    return


@dataloader.command()
@get_config_path_option()
def params(config_path: str) -> None:
    """显示用于数据加载器选数的可用参数。"""
    mod = get_module(config_path)
    dataloader_cls = get_dataloader_cls(mod)
    if dataloader_cls is None:
        return
    all_params = dataloader_cls.get_all_params()
    cols = list({param for params in all_params for param in params})
    available_params = pd.DataFrame({col: [params.get(col, '-') for params in all_params] for col in cols})
    print_console([available_params], ['可用参数'])


@dataloader.command()
@get_config_path_option()
def odds_types(config_path: str) -> None:
    """显示可用于提取赔率数据的赔率类型。"""
    mod = get_module(config_path)
    dataloader_cls = get_dataloader_cls(mod)
    if dataloader_cls is None:
        return
    param_grid = get_param_grid(mod)
    odds_types = pd.DataFrame(dataloader_cls(param_grid).get_odds_types(), columns=['类型'])
    print_console([odds_types], ['可用赔率类型'])


@dataloader.command()
@get_config_path_option()
@get_data_path_option()
def training(config_path: str, data_path: str) -> None:
    """使用数据加载器提取训练数据。"""
    mod = get_module(config_path)
    dataloader_cls = get_dataloader_cls(mod)
    if dataloader_cls is None:
        return
    param_grid = get_param_grid(mod)
    drop_na_thres = get_drop_na_thres(mod)
    odds_type = get_odds_type(mod)
    dataloader = dataloader_cls(param_grid)
    X_train, Y_train, O_train = dataloader.extract_train_data(drop_na_thres=drop_na_thres, odds_type=odds_type)
    print_console(
        [X_train, Y_train] + ([O_train] if O_train is not None else []),
        ['训练输入数据', '训练输出数据'] + (['训练赔率数据'] if O_train is not None else []),
    )
    if data_path is not None:
        (Path(data_path) / 'sports-betting-data').mkdir(parents=True, exist_ok=True)
        X_train.to_csv(Path(data_path) / 'sports-betting-data' / 'X_train.csv')
        Y_train.to_csv(Path(data_path) / 'sports-betting-data' / 'Y_train.csv')
        if O_train is not None:
            O_train.to_csv(Path(data_path) / 'sports-betting-data' / 'O_train.csv')


@dataloader.command()
@get_config_path_option()
@get_data_path_option()
def fixtures(config_path: str, data_path: str) -> None:
    """使用数据加载器提取赛程数据。"""
    console = Console()
    mod = get_module(config_path)
    dataloader_cls = get_dataloader_cls(mod)
    if dataloader_cls is None:
        return
    param_grid = get_param_grid(mod)
    drop_na_thres = get_drop_na_thres(mod)
    odds_type = get_odds_type(mod)
    dataloader = dataloader_cls(param_grid)
    dataloader.extract_train_data(drop_na_thres=drop_na_thres, odds_type=odds_type)
    X_fix, _, O_fix = dataloader.extract_fixtures_data()
    if not X_fix.empty:
        print_console([X_fix], ['赛程输入数据'])
        if O_fix is not None and not O_fix.empty:
            print_console([O_fix], ['赛程赔率数据'])
        if data_path is not None:
            (Path(data_path) / 'sports-betting-data').mkdir(parents=True, exist_ok=True)
            X_fix.to_csv(Path(data_path) / 'sports-betting-data' / 'X_fix.csv')
            if O_fix is not None and not O_fix.empty:
                O_fix.to_csv(Path(data_path) / 'sports-betting-data' / 'O_fix.csv')
    else:
        warning = Panel.fit(
            '[bold red]赛程数据为空',
        )
        console.print(warning)
