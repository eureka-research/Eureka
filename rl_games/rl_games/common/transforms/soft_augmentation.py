from rl_games.common.transforms import transforms
import torch

class SoftAugmentation():
    def __init__(self, **kwargs):
        self.transform_config = kwargs.pop('transform')
        self.aug_coef = kwargs.pop('aug_coef', 0.001)
        print('aug coef:', self.aug_coef)
        self.name = self.transform_config['name']

        #TODO: remove hardcode
        self.transform = transforms.ImageDatasetTransform(**self.transform_config)

    def get_coef(self):
        return self.aug_coef

    def get_loss(self, p_dict, model, input_dict, loss_type = 'both'):
        '''
        loss_type: 'critic', 'policy', 'both'
        '''
        if self.transform:
            input_dict = self.transform(input_dict)
        loss = 0
        q_dict = model(input_dict)
        if loss_type == 'policy' or loss_type == 'both':
            p_dict['logits'] = p_dict['logits'].detach()
            loss = model.kl(p_dict, q_dict)
        if loss_type == 'critic' or loss_type == 'both':
            p_value = p_dict['value'].detach()
            q_value = q_dict['value']
            loss = loss + (0.5 * (p_value - q_value)**2).sum(dim=-1)
        
        return loss