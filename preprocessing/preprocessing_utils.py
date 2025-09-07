import yaml
from pathlib import Path

class BackupBase:
    def __init__(self):
        self.data_dict = {}

    def save_with_key(self, data, key, copy=False):
        if key in self.data_dict.keys():
            print(f'Data Key "f{key}" already exists, Skip Saving')
        else:
            self.data_dict[key] = data.copy() if copy else data

    def check_key(self, key):
        return key in self.data_dict.keys()

    def copy_data_with_key(self, key):
        return self.data_dict[key].copy()
    
Backup = BackupBase()

def get_data_paths(config_path=None):
    base_yaml_path = Path(config_path, 'base.yaml') if config_path else Path('../configs', 'base.yaml')
    with open(base_yaml_path, 'r', encoding='utf-8') as file:
        base_config = yaml.safe_load(file)

    asset_path = base_config['paths']['asset_path']
    data_path  = base_config['paths']['data_path']

    return asset_path, data_path

def get_figure_path(config_path=None):
    base_yaml_path = Path(config_path, 'base.yaml') if config_path else Path('../configs', 'base.yaml')
    with open(base_yaml_path, 'r', encoding='utf-8') as file:
        base_config = yaml.safe_load(file)

    figure_path = base_config['paths']['figure_path']

    return figure_path