import os
import pandas as pd
from scipy.interpolate import interp1d
import numpy as np


class ExperimentalAPSet:
    """ Object containing iPSC-CM APs recorded during the dynamic clamp portion of
    the dynamically-rich protocol (2021).
    Attributes:
      AP_set: A dict containing single action potentials from experimental
              dynamic-clamp recording.
      score(model_AP_set): function for evaluating model fitness.
    """

    def __init__(self, path, dc_ik1, file_prefix='cell_', file_suffix='.txt',
                 cell_id=0):
        self.path = path
        self.file_prefix = file_prefix
        self.file_suffix = file_suffix
        self.cell_id = cell_id # optional identifier for organization
        self.dc_ik1 = dc_ik1 # scaling coefficients on Ishihara IK1
        filenames = os.listdir(path)

        """Data is formatted so that each file contains an single AP waveform.
         The waveform is assumed to be a 2 column file with time and voltage.
         The point t(0) is set to the dV/dt_max of the recording. This allows
         the simulated APs to align at the t(0) = dV/dt_max.
        """
        self.AP_set = {}
        for i in filenames:
            # Checks that file prefix and suffix are identified.
            if (i.find(file_prefix) >= 0 and i.find(file_suffix) >= 0):
                tmp_key = i[len(file_prefix):i.find(file_suffix)]
                tmp_ap = pd.read_csv(path+'/'+i, delimiter=' ')
                self.AP_set[tmp_key] = tmp_ap
            else:
                print('Could not local file(s). Check file(s) and/or directory.')
                print('path: '+self.path)
                print('filename: '+i)

    def get_info(self):
        info_string = 'cell_id: '+str(self.cell_id)+'/n'
        info_string += 'dir: '+self.path+'/n'
        info_string += 'meta: '+self.meta+'/n'
        print(info_string)

    def get_AP_set(self):
        return self.AP_set

    def score(self, model_AP_set, model_id=0, write_data=False):
        # Score assigned if there was an AP Failure
        MAX_SCORE = 1000.0
        
        scores = {}
        ap_keys = list(self.AP_set.keys())

        # Check for AP Failure
        if (model_AP_set[1]):
            for i in ap_keys:
                scores[i] = MAX_SCORE
            return scores
        for i in ap_keys:
            try:
                real = self.AP_set[i]
                simu = model_AP_set[0][i]
                # Find time series boundaries
                t_first_real = round(real.iat[0, 0], 1)
                nrows_real = real.shape[0]
                t_last_real = round(real.iat[(nrows_real-1), 0], 1)
                t_resolution = round(real.iat[1, 0], 1) - round(real.iat[0, 0], 1)
                t_resolution = round(t_resolution, 1)
                t_first_simu = round(simu.iat[0, 0], 1)
                nrows_simu = simu.shape[0]
                t_last_simu = round(simu.iat[(nrows_simu-1), 0], 1)

                # Align curves within interpolation bounds
                if (t_first_real >= t_first_simu):
                    t_first = t_first_real + t_resolution
                else:
                    t_first = t_first_simu + t_resolution

                if (t_last_real >= t_last_simu):
                    t_last = t_last_simu - t_resolution
                else:
                    t_last = t_last_real - t_resolution
                N = int((t_last - t_first)/t_resolution)
                t_new = np.linspace(t_first, t_last, N)

                f_simu = interp1d(simu.iloc[:, 0], simu.iloc[:, 1])
                mV_new_simu = f_simu(t_new)
                f_real = interp1d(real.iloc[:, 0], real.iloc[:, 1])
                mV_new_real = f_real(t_new)

                # Calculate Root Mean Square Error
                n = float(len(t_new))
                rmse = (sum((mV_new_real - mV_new_simu)**2) / n)**0.5
                scores[i] = rmse

                # Write AP files
                if write_data:
                    d = {'t':t_new, 'mV_cell':mV_new_real, 'mV_simu':mV_new_simu}
                    d = pd.DataFrame(d)
                    filename = self.file_prefix + i + '_scored_AP_'+str(model_id)+'.txt'
                    d.to_csv(filename, sep=' ', index=False)
            except KeyError:
                print('Model AP_set keys did not match ExperimentalAPSet keys.')
        return scores
