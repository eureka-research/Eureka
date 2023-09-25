

'''
TODO: move play_steps here
'''
class Rollout:
    def __init__(self, gamma):
        self.gamma = gamma

    def play_steps(self, env, max_steps_count = 1):
        pass


class DiscretePpoRollout(Rollout):
    def __init__(self, gamma, lam):
        super(Rollout, self).__init__(gamma)
        self.lam = lam

    def play_steps(self, env, max_steps_count = 1):
        pass