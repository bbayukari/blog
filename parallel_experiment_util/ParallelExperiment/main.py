import pandas as pd
import multiprocessing as mp
import numpy as np
from util import product_dict

class ParallelExperiment:
    def __init__(self, task, in_keys, out_keys, processes=1, name="test"):
        """
        in_keys:
            in_keys are arrays of strings which will be the keys of task's in_para.
        out_keys:
            out_keys are arrays of strings which will be the keys of task's out_para.
        """
        self.task = task
        self.in_keys = in_keys
        self.out_keys = out_keys
        assert len(set(self.in_keys) & set(self.out_keys)) == 0
        self.name = name
        self.results = []  # 'results' is a list of dict
        self.processes = processes

    def task_parallel(self, para):
        try:
            result = self.task(**para)
        except Exception as e:
            print(e)
            result = {para: np.nan for para in self.out_keys}
        para.update(result)
        print(para, flush=True)
        return para

    def check(self, **in_para):
        if set(in_para.keys()) != set(self.in_keys):
            raise RuntimeError("in_parameter's keys do not match!\n{}\n{}".format(self.in_keys, set(in_para.keys())))

        result = self.task(**in_para)
        print(result)

        if not isinstance(result, dict) or set(result.keys()) != set(self.out_keys):
            raise RuntimeError("out_parameter's keys do not match!\n{}\n{}".format(self.out_keys, set(result.keys())))

    def run(self, in_para_list = None, /, **in_para):
        """
        in_para.keys should be the same as self.in_keys and in_para.values should be arrays.
        """
        if in_para_list is None:
            in_para_list = product_dict(**in_para)

        with mp.Pool(processes=self.processes) as pool:
            self.results.extend(
                pool.starmap(
                    self.task_parallel, [(para,) for para in in_para_list]
                ),
            )

    def save(self, filename=None):
        if filename is None:
            filename = self.name

        pd.DataFrame(
            {
                para: [result[para] for result in self.results]
                for para in self.in_keys + self.out_keys
            }
        ).to_csv(filename + ".csv")
