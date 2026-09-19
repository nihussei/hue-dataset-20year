import numpy as np
import pandas as pd

class Prosumer:
    def __init__(self, df: pd.DataFrame):
        self.name = f'P{df["House"].iat[0]}'
        df = df.rename(columns={'energy_kWh':'load', 'ac_output':'pv'}).reset_index(drop=True)
        self.timeData = df
        self.timeData['net_load'] = np.maximum(self.timeData['load']-self.timeData['pv'], 0) # (Eq. 6)
        self.timeData['contribution'] = np.maximum(self.timeData['pv']-self.timeData['load'], 0) # (Eq. 1)
        self.timeData['accepted_pv'] = 0.0
        self.timeData['energy_allocated'] = 0.0
        self.timeData['contribution_index'] = 1.0
        self.priority_weight = np.nan
        self.mu = np.nan # fractional share of BESS operating cost
        self.gamma = np.nan # fractional share of BESS degradation
        self.C_repl = np.nan # share of BESS replacement cost
        self.total_net_cost = np.nan
        
    def calculate_Delta_f(self, battery): # TODO: consider if this should calculate one time step at a time
        T = battery.timeData['T_c']
        sigma = battery.timeData['SoC']
        S_sigma = np.exp(battery.k_sigma*(sigma-battery.sigma_ref))                                   # Eq. 17
        S_T = np.exp(battery.k_T*(T-battery.T_ref)*(battery.T_ref/T))                                 # Eq. 16
        s_delta = battery.calculate_dS_ddelta()
        
        self.timeData["delta_f"] = (S_sigma*S_T*s_delta
                                    *(self.timeData['accepted_pv']+self.timeData['energy_allocated'])
                                    /battery.timeData['capacity'])                                      # Eq. 23
        
    def calculate_C_repl(self, battery, Y_op, Y_base, Y_plan):
        C_repl_op = battery.calculate_C_repl_op(Y_op, Y_base, Y_plan)
        self.C_repl = self.gamma*C_repl_op                                                              # Eq. 36
        self.timeData['C_repl'] = self.C_repl/len(self.timeData)