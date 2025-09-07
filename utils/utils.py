from pathlib import Path
import yaml
import argparse
import logging
import os
import random
import numpy as np
import pandas as pd
import sys

def _get_config(config_name):
    config_path = Path('./configs', f"{config_name}.yaml")
    if not config_path.exists():
        raise ValueError(f"Config file not found at {config_path}")

    # Load Config file
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    return config

def merge_configs(base_config, override_config):
    for key, value in override_config.items():
        if isinstance(value, dict) and key in base_config:
            base_config[key] = merge_configs(base_config.get(key, {}), value)
        else:
            base_config[key] = value
    return base_config

def get_configs(base_config_name):
    args = argparse.ArgumentParser()
    args.add_argument('--config', type=str)
    args.add_argument('--device', type=str)
    args.add_argument('--train_total', action='store_true')
    args = args.parse_args()

    base_config = _get_config(base_config_name)
    if args.config:
        override_config = _get_config(args.config)
        config = merge_configs(base_config, override_config)
    else:
        config = base_config

    # Set device
    if args.device:
        config['device'] = args.device
    
    config['data'] = config.get('data', {})
    config['data']['train_total'] = args.train_total or config.get('data', {}).get('train_total', False)

    return config

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Python RNG
    np.random.seed(seed)
    random.seed(seed)

def get_logger(name, output_path, add_stream_handler=False):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.hasHandlers():
        # File handler
        file_handler = logging.FileHandler(output_path / 'train.log')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

        # Stream handler
        if add_stream_handler:
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            stream_handler.setLevel(logging.INFO)
            logger.addHandler(stream_handler)

        logger.info('Logger initialized.')

    return logger

def set_output_path(config, sub=None):
# Set Output Path
    y, m, d, H, M, S = pd.Timestamp.now().timetuple()[:6]
    y = str(y)[2:]
    output_path = Path(config['paths']['output_path'], sub) if sub else Path(config['paths']['output_path'])
    output_path = Path(output_path, f"{y}-{m}-{d}", f"{H}-{M}-{S}")

    print(f'>>> Output Path: {output_path} <<<')
    output_path.mkdir(parents=True, exist_ok=True)
    
    # YAML dump with extra newline between main sections
    with open(output_path / 'config.yaml', 'w', encoding='utf-8') as file:
        yaml_str = yaml.dump(config, default_flow_style=False, sort_keys=False)
        formatted_yaml_str = insert_newlines_between_main_sections(yaml_str)
        file.write(formatted_yaml_str)
    print()

    return output_path

def insert_newlines_between_main_sections(yaml_str):
    lines = yaml_str.splitlines()
    formatted_lines = []
    for i, line in enumerate(lines):
        formatted_lines.append(line)
        # Add a newline before top-level keys
        if i < len(lines) - 1 and lines[i+1] and not lines[i+1].startswith(' '):
            formatted_lines.append('')
    return '\n'.join(formatted_lines)