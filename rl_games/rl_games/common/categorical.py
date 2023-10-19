import numpy as np


class CategoricalQ:
    def __init__(self, n_atoms, v_min, v_max):
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (n_atoms - 1)

    def distr_projection(self, next_distr, rewards, dones, gamma):
        """
        Perform distribution projection aka Catergorical Algorithm from the
        "A Distributional Perspective on RL" paper
        """
        proj_distr = np.zeros_like(next_distr, dtype=np.float32)
        n_atoms = self.n_atoms
        v_min = self.v_min
        v_max = self.v_max
        delta_z = self.delta_z
        for atom in range(n_atoms):
            z = rewards + (v_min + atom * delta_z) * gamma
            tz_j = np.clip(z, v_min, v_max)
            b_j = (tz_j - v_min) / delta_z
            l = np.floor(b_j).astype(np.int64)
            u = np.ceil(b_j).astype(np.int64)
            eq_mask = u == l
            proj_distr[eq_mask, l[eq_mask]] += next_distr[eq_mask, atom]
            ne_mask = u != l
            proj_distr[ne_mask, l[ne_mask]] += next_distr[ne_mask, atom] * (u - b_j)[ne_mask]
            proj_distr[ne_mask, u[ne_mask]] += next_distr[ne_mask, atom] * (b_j - l)[ne_mask]
            
        if dones.any():
            proj_distr[dones] = 0.0
            tz_j = np.clip(rewards[dones], v_min, v_max)

            b_j = (tz_j - v_min) / delta_z
            l = np.floor(b_j).astype(np.int64)
            u = np.ceil(b_j).astype(np.int64)
            eq_mask = u == l
            eq_dones = dones.copy()
            eq_dones[dones] = eq_mask
            if eq_dones.any():
                proj_distr[eq_dones, l[eq_mask]] = 1.0
            ne_mask = u != l
            ne_dones = dones.copy()
            ne_dones[dones] = ne_mask
            if ne_dones.any():
                proj_distr[ne_dones, l[ne_mask]] = (u - b_j)[ne_mask]
                proj_distr[ne_dones, u[ne_mask]] = (b_j - l)[ne_mask]
        return proj_distr