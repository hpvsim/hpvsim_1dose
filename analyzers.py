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

    @staticmethod
    def reduce(analyzers, use_mean=False, quantiles=None):
        # Process quantiles
        if quantiles is None:
            quantiles = {'low':0.1, 'high':0.9}
        if not isinstance(quantiles, dict):
            try:
                quantiles = {'low':float(quantiles[0]), 'high':float(quantiles[1])}
            except Exception as E:
                errormsg = f'Could not figure out how to convert {quantiles} into a quantiles object: must be a dict with keys low, high or a 2-element array ({str(E)})'
                raise ValueError(errormsg)

        # Get base analyzer properties and copy them into reduced analyzer
        base_analyzer = analyzers[0]
        reduced_analyzer = sc.dcp(base_analyzer)
        ashape = base_analyzer.results.shape  # Figure out dimensions
        new_ashape = ashape + (len(analyzers),)
        raw = np.zeros(new_ashape)

        # Pull out results for each analyzer
        for a, analyzer in enumerate(analyzers):
            raw[:, a] = analyzer.results

        # Get quantiles
        reduced_analyzer.results.best = np.quantile(raw, q=0.5, axis=-1)
        reduced_analyzer.results.low  = np.quantile(raw, q=quantiles['low'], axis=-1)
        reduced_analyzer.results.high = np.quantile(raw, q=quantiles['high'], axis=-1)

        return reduced_analyzer
        