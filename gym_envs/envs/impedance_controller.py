# MODEL: Impedance controller (impedance / admittance)
# AUTHOR: Yi Liu @AiRO 
# UNIVERSITY: UGent-imec
# DEPARTMENT: Faculty of Engineering and Architecture
# Control Engineering / Automation Engineering

import numpy as np

class FT_controller():
    def __init__(self, m, b, k, dt, tau_m, tau_b, tau_k) -> None:
        self.M = m
        self.B = b
        self.K = k
        self.dt = dt
        self._mat = np.zeros((3, 3))
        self.T_mat = np.zeros((3, 3))

        self.TM = tau_m
        self.TB = tau_b
        self.TK = tau_k

    def admittance_control(self, desired_position, desired_rotation,
                           FT_data, 
                           params_mat, paramsT_mat) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        F_x = FT_data[0]
        F_y = FT_data[1]
        F_z = FT_data[2]
        T_x = FT_data[3]
        T_y = FT_data[4]
        T_z = FT_data[5]

        # F_x = 0
        # F_y = 0
        # F_z = 0
        # print(F_x, F_y, F_z)

        ddx_e = params_mat[0, 0]
        dx_e = params_mat[0, 1]
        x_e = params_mat[0, 2]
        ddy_e = params_mat[1, 0]
        dy_e = params_mat[1, 1]
        y_e = params_mat[1, 2]
        ddz_e = params_mat[2, 0]
        dz_e = params_mat[2, 1]
        z_e = params_mat[2, 2]

        ddx_e_d = (F_x - self.B * dx_e - self.K * x_e) / self.M
        dx_e_d = self.dt * ddx_e_d + dx_e
        x_e_d = self.dt * dx_e_d + x_e
        
        ddy_e_d = (F_y - self.B * dy_e - self.K * y_e) / self.M
        dy_e_d = self.dt * ddy_e_d + dy_e
        y_e_d = self.dt * dy_e_d + y_e

        ddz_e_d = (F_z - self.B * dz_e - self.K * z_e) / self.M
        dz_e_d = self.dt * ddz_e_d + dz_e
        z_e_d = self.dt * dz_e_d + z_e

        # position_offset = np.array([0, 0, 0])
        position_offset = np.array([x_e_d*0.1, y_e_d*0.1, z_e_d])
        # print("---", F_z, ddz_e, dz_e, z_e, ddz_e_d, dz_e_d, z_e_d)
        
        position_d = position_offset + desired_position


        ddTx_e = paramsT_mat[0, 0]
        dTx_e = paramsT_mat[0, 1]
        Tx_e = paramsT_mat[0, 2]
        ddTy_e = paramsT_mat[1, 0]
        dTy_e = paramsT_mat[1, 1]
        Ty_e = paramsT_mat[1, 2]
        ddTz_e = paramsT_mat[2, 0]
        dTz_e = paramsT_mat[2, 1]
        Tz_e = paramsT_mat[2, 2]

        ddTx_e_d = (T_x - self.TB * dTx_e - self.TK * Tx_e) / self.TM
        dTx_e_d = self.dt * ddTx_e_d + dTx_e
        Tx_e_d = self.dt * dTx_e_d + Tx_e

        ddTy_e_d = (T_y - self.TB * dTy_e - self.TK * Ty_e) / self.TM
        dTy_e_d = self.dt * ddTy_e_d + dTy_e
        Ty_e_d = self.dt * dTy_e_d + Ty_e

        ddTz_e_d = (T_z - self.TB * dTz_e - self.TK * Tz_e) / self.TM
        dTz_e_d = self.dt * ddTz_e_d + dTz_e
        Tz_e_d = self.dt * dTz_e_d + Tz_e

        rotation_offset = np.array([Tx_e_d, Ty_e_d, Tz_e_d]) * 0.1
        rotation_d = rotation_offset + desired_rotation
        # print("rotation_offset:", rotation_offset)
        # print("rotation_d:", rotation_d)
        # print("FT:", FT_data)

        self._mat[0, 0] = ddx_e_d
        self._mat[0, 1] = dx_e_d
        self._mat[0, 2] = x_e_d
        self._mat[1, 0] = ddy_e_d
        self._mat[1, 1] = dy_e_d
        self._mat[1, 2] = y_e_d
        self._mat[2, 0] = ddz_e_d
        self._mat[2, 1] = dz_e_d
        self._mat[2, 2] = z_e_d

        self.T_mat[0, 0] = ddTx_e_d
        self.T_mat[0, 1] = dTx_e_d
        self.T_mat[0, 2] = Tx_e_d
        self.T_mat[1, 0] = ddTy_e_d
        self.T_mat[1, 1] = dTy_e_d
        self.T_mat[1, 2] = Ty_e_d
        self.T_mat[2, 0] = ddTz_e_d
        self.T_mat[2, 1] = dTz_e_d
        self.T_mat[2, 2] = Tz_e_d

        return position_d, rotation_d, self._mat, self.T_mat

