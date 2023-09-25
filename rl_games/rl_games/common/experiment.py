import copy
import yaml

class Experiment:
    def __init__(self, config, experiment_config):
        self.config = copy.deepcopy(config)
        self.best_config = copy.deepcopy(self.config)
        self.experiment_config = experiment_config
        self.best_results = -100500, 0
        self.use_best_prev_result = self.experiment_config.get('use_best_prev_result', True)

        self.experiments = self.experiment_config['experiments']

        self.last_exp_idx = self.experiment_config.get('start_exp', 0)
        self.sub_idx = self.experiment_config.get('start_sub_exp', 0)

        self.done = False
        self.results = {}
        self.create_config()

    def _set_parameter(self, config, path, value):
        keys = path.split('.')
        sub_conf = config
        for key in keys[:-1]:
            sub_conf = sub_conf[key]
        print('set:' + str(keys) + ':' + str(value))
        sub_conf[keys[-1]] = value

    def set_results(self, rewards, epochs):
        self.results[(self.last_exp_idx, self.sub_idx)] = rewards, epochs
        if self.best_results[0] < rewards:
            self.best_results = rewards, epochs

    def create_config(self):
        if self.done:
            self.current_config = None
            return
        self.current_config = copy.deepcopy(self.config)
        self.current_config['config']['name'] += '_' + str(self.last_exp_idx) + '_' + str(self.sub_idx)
        print('Experiment name: ' + self.current_config['config']['name'])
        for key in self.experiments[self.last_exp_idx]['exp']:
            self._set_parameter(self.current_config, key['path'], key['value'][self.sub_idx])

        with open('data.yml', 'w') as outfile:
            yaml.dump(self.current_config, outfile, default_flow_style=False)

    def get_next_config(self):
        config = self.current_config
        max_vals = len(self.experiments[0]['exp'][0]['value'])
        self.sub_idx += 1
        if self.sub_idx >= max_vals:
            self.sub_idx = 0
            self.last_exp_idx += 1
            if self.last_exp_idx >= len(self.experiments):
                self.done = True
            else:
                self.last_exp_idx += 1

        self.create_config()
        return config

    #def __iter__(self):
    #    print('__iter__')
    #    return self

    def __next__(self):
        print('__next__')
        res = self.get_next_config()
        if res is not None:
            yield res
