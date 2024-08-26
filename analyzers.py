"""
Define custom analyzers
"""

import numpy as np
import pandas as pd
import sciris as sc
import hpvsim as hpv
import hpvsim.utils as hpu


class cohort_cancers(hpv.Analyzer):
    def __init__(self, cohort_age=None, start=None, **kwargs):
        super().__init__(**kwargs)
        self.start = start or 2023
        self.cohort_age = cohort_age or [9, 14]
        self.years = None

        return

    def initialize(self, sim):
        super().initialize()
        self.si = sc.findfirst(sim.res_yearvec, self.start)
        self.npts = len(sim.res_yearvec[self.si:])
        self.years = sim.res_yearvec[self.si:]
        self.results = np.zeros(self.npts)
        return

    def apply(self, sim):
        if sim.yearvec[sim.t] >= self.start:
            li = np.floor(sim.yearvec[sim.t])
            idx = sc.findfirst(self.years, li)
            ppl = sim.people

            time_elapsed = sim.yearvec[sim.t] - self.start
            current_age_range = [self.cohort_age[0]+time_elapsed, self.cohort_age[1]+time_elapsed]

            cic = (ppl.date_cancerous == sim.t) & (ppl.age >= current_age_range[0]) & (ppl.age <= current_age_range[1])
            if cic.any():
                self.results[idx] += sum(ppl.scale[hpu.true(cic)])

        return


