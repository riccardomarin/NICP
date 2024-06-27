from pathlib import Path
from typing import Union
import sys

import tomlkit


def modify_path(config: dict):
    sys.path.append(config['AMASS_REPO'])
    sys.path.append(str(Path(config['AMASS_REPO']) / 'src/amass'))
    sys.path.append(str(Path(config['HUMAN_BODY_PRIOR_REPO']) / "src"))


def load_config(config_path: Union[Path, str]) -> dict:
    config_path = Path(config_path)
    config = dict(tomlkit.parse(config_path.read_text()))

    config['SMPL_DIR'] = Path(config['SMPL_DIR'])
    if not config['SMPL_DIR'].is_absolute():
        config['SMPL_DIR'] = Path(config["AMASS_REPO"]) / config['SMPL_DIR']

    config["TARGET_DATA"] = Path(config["TARGET_DATA"])

    return config
