class IVecEnv:
    def step(self, actions):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def has_action_masks(self):
        return False

    def get_number_of_agents(self):
        return 1

    def get_env_info(self):
        pass

    def seed(self, seed):
        pass

    def set_train_info(self, env_frames, *args, **kwargs):
        """
        Send the information in the direction algo->environment.
        Most common use case: tell the environment how far along we are in the training process. This is useful
        for implementing curriculums and things such as that.
        """
        pass

    def get_env_state(self):
        """
        Return serializable environment state to be saved to checkpoint.
        Can be used for stateful training sessions, i.e. with adaptive curriculums.
        """
        return None

    def set_env_state(self, env_state):
        pass
