"""Module that contains the main function of the CLI."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import click

from ._betting import bettor
from ._data import dataloader


@click.group()
def main() -> None:
    """sports-betting 的命令行工具。

    当你输入 `sportsbet` 或 `python -m sportsbet` 时会执行该命令。
    """
    return


main.add_command(dataloader)
main.add_command(bettor)
