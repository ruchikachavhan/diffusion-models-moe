import os
import yaml


class Config:
    def __init__(self, path, exp_name):
        # Load config file
        with open(path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        self.config = config
        self.exp_name = exp_name
        for key, value in config.items():
            setattr(self, key, value)
        
        self.makedirs()

    def makedirs(self):
        self.save_path = os.path.join('compare_models_results', self.save_path)
        self.res_path = self.save_path  
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
