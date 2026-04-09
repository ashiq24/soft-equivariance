import os
from typing import Dict, Optional

import wandb


def read_wandb_key(keys_path: str) -> Optional[str]:
    if not os.path.isfile(keys_path):
        return None
    try:
        with open(keys_path, 'r') as f:
            return f.read().strip()
    except Exception:
        return None


def init_wandb(cfg: Dict, keys_path: str = './keys/wandb.txt', run_name: Optional[str] = None, dir: str = './logs'):
    key = read_wandb_key(keys_path)
    if key:
        os.environ['WANDB_API_KEY'] = key

    project = cfg.get('experiment', {}).get('project', 'soft-equivariance')
    name = run_name or cfg.get('experiment', {}).get('run_name', None)

    run = wandb.init(project=project, name=name, config=cfg, dir=dir)

    def logger(metrics: Dict):
        wandb.log(metrics)

    return logger, run


