import numpy as np

class SelfPlayManager:
    def __init__(self, config, writter):
        self.config = config
        self.writter = writter
        self.update_score = self.config['update_score']
        self.games_to_check = self.config['games_to_check']
        self.check_scores = self.config.get('check_scores', False)
        self.env_update_num = self.config.get('env_update_num', 1)
        self.env_indexes = np.arange(start=0, stop=self.env_update_num)
        self.updates_num = 0
        
    def update(self, algo):
        self.updates_num += 1
        if self.check_scores:
            data = algo.game_scores
        else:
            data = algo.game_rewards

        if len(data) >= self.games_to_check:
            mean_scores = data.get_mean()
            mean_rewards = algo.game_rewards.get_mean()
            if mean_scores > self.update_score:
                print('Mean scores: ', mean_scores, ' mean rewards: ', mean_rewards, ' updating weights')

                algo.clear_stats()
                self.writter.add_scalar('selfplay/iters_update_weigths', self.updates_num, algo.frame)
                algo.vec_env.set_weights(self.env_indexes, algo.get_weights())
                self.env_indexes = (self.env_indexes + 1) % (algo.num_actors)
                self.updates_num = 0
                      