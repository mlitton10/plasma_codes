

class TwoFluidPlasma:
    def __init__(self, B, m_i, q_i, n, T):
        self.B_gauss = B
        self.m_ion = m_i
        self.q_ion = q_i

        self.m_electron = 0.0
        self.q_electron = 0.0

        self.n_cm = n
        self.T_ev = T

        self.B_tesla, self.n_meters = self._convert_cgs_to_mks()

    def _convert_cgs_to_mks(self):
        B_tesla = self.B_gauss * 1e-4
        n_m = self.n_cm * 1e-6

        return B_tesla, n_m

