import torch
import rl_games.algos_torch.torch_ext as torch_ext

class DefaultDiagnostics(object):
    def __init__(self):
        pass
    def send_info(self, writter):
        pass    
    def epoch(self, agent, current_epoch):
        pass
    def mini_epoch(self, agent, miniepoch):
        pass
    def mini_batch(self, agent, batch, e_clip, minibatch):
        pass


class PpoDiagnostics(DefaultDiagnostics):
    def __init__(self):
        self.diag_dict = {}
        self.clip_fracs = []
        self.exp_vars = []
        self.current_epoch = 0

    def send_info(self, writter):
        if writter is None:
            return
        for k,v in self.diag_dict.items():
            writter.add_scalar(k, v.cpu().numpy(), self.current_epoch)
    
    def epoch(self, agent, current_epoch):
        self.current_epoch = current_epoch
        if agent.normalize_rms_advantage:
            self.diag_dict['diagnostics/rms_advantage/mean'] = agent.advantage_mean_std.moving_mean
            self.diag_dict['diagnostics/rms_advantage/var'] = agent.advantage_mean_std.moving_var
        if agent.normalize_value:
            self.diag_dict['diagnostics/rms_value/mean'] = agent.value_mean_std.running_mean
            self.diag_dict['diagnostics/rms_value/var'] = agent.value_mean_std.running_var

        exp_var = torch.stack(self.exp_vars, axis=0).mean()
        self.exp_vars = []
        self.diag_dict['diagnostics/exp_var'] = exp_var

    def mini_epoch(self, agent, miniepoch):
        clip_frac = torch.stack(self.clip_fracs, axis=0).mean()
        self.clip_fracs = []
        self.diag_dict['diagnostics/clip_frac/{0}'.format(miniepoch)] = clip_frac


    def mini_batch(self, agent, batch, e_clip, minibatch):
        with torch.no_grad():
            values = batch['values'].detach()
            returns = batch['returns'].detach()
            new_neglogp = batch['new_neglogp'].detach()
            old_neglogp = batch['old_neglogp'].detach()
            masks = batch['masks']
            exp_var = torch_ext.explained_variance(values, returns, masks)

            clip_frac = torch_ext.policy_clip_fraction(new_neglogp, old_neglogp, e_clip, masks)
            self.exp_vars.append(exp_var)
            self.clip_fracs.append(clip_frac)
            

            

