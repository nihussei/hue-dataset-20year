"""Reproduce Client Revision 01 with fixed validated D3QN models.

Usage:
    python scripts/run_client_revision_01.py --hue-root /path/to/HUE --output revision01_results

The script regenerates the deterministic 2021-2040 HUE augmentation, fits the
constrained balanced tariff assignment on the training years only, holds the
validated D3QN model weights fixed, and evaluates ToU + Balanced for four
controllers on held-out 2040.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd

from microgrid_opt.augmentation import AugmentationConfig, prepare_augmented_hue_community
from microgrid_opt.config import ExperimentConfig
from microgrid_opt.controllers import D3QNController, MPCController, RuleBasedController
from microgrid_opt.core import MicrogridEnvironment
from microgrid_opt.data import chronological_year_split
from microgrid_opt.evaluation import run_controller
from microgrid_opt.metrics import (disaggregated_metrics, prosumer_metrics,
                                   reliability_by_prosumer, temporal_metrics)
from microgrid_opt.plots import plot_controller_comparison
from microgrid_opt.tariffs import fit_balanced_tariff_assignments
from microgrid_opt.validation import validate_hourly_result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--hue-root',required=True)
    ap.add_argument('--output',default='client_revision_01_results')
    args=ap.parse_args()
    root=Path(__file__).resolve().parents[1]
    out=Path(args.output); (out/'hourly').mkdir(parents=True,exist_ok=True); (out/'tables').mkdir(exist_ok=True); (out/'figures').mkdir(exist_ok=True)
    data,_=prepare_augmented_hue_community(args.hue_root,10,augmentation=AugmentationConfig(start_year=2021,years=20,seed=42),outage_scenario='moderate')
    train,_,test,_=chronological_year_split(data,2,1)
    cfg=ExperimentConfig()
    fixed,fit=fit_balanced_tariff_assignments(train,cfg.tariff)
    fit.to_csv(out/'tables/balanced_tariff_assignments_from_training.csv',index=False)
    controllers=[RuleBasedController(),MPCController(cfg.mpc_horizon_hours),
                 D3QNController.load(str(root/'pretrained_models/d3qn_per_model.npz'),name='d3qn_per'),
                 D3QNController.load(str(root/'pretrained_models/d3qn_uniform_replay_model.npz'),name='d3qn_uniform_replay')]
    temps=[]; pros=[]; dis=[]; checks=[]; rows=[]
    for tariff in ('tou','balanced'):
        for controller in controllers:
            env=MicrogridEnvironment(test,deepcopy(cfg),tariff=tariff,fixed_tariffs=fixed)
            r=run_controller(env,controller).assign(controller=controller.name,tariff=tariff,allocation_method='proportional')
            r.to_csv(out/'hourly'/f'{controller.name}_{tariff}.csv.gz',index=False,compression='gzip')
            rel=reliability_by_prosumer(r,env.ids); affected=int((rel.SAIFI_interruptions>0).sum())
            rows.append({'controller':controller.name,'tariff':tariff,
                         'cumulative_reward':float(r.reward.sum()),
                         'grid_cost_cad':float(r.grid_cost_cad.sum()),
                         'calendar_degradation_cost_cad':float(r.calendar_degradation_cost_cad.sum()),
                         'cycle_degradation_cost_cad':float(r.cycle_degradation_cost_cad.sum()),
                         'degradation_cost_cad':float(r.degradation_cost_cad.sum()),
                         'net_cost_cad':float((r.grid_cost_cad+r.degradation_cost_cad).sum()),
                         'pv_utilization':float(np.average(r.pv_utilization,weights=np.maximum(r.community_pv_kwh,1e-9))),
                         'unserved_kwh':float(r.unserved_kwh.sum()),
                         'SAIDI_hours':float(rel.SAIDI_hours.mean()),
                         'SAIFI_interruptions':float(rel.SAIFI_interruptions.mean()),
                         'CAIFI_interruptions_affected_customer':float(rel.SAIFI_interruptions.sum()/affected) if affected else 0.0})
            checks.append({'controller':controller.name,'tariff':tariff,**validate_hourly_result(r,cfg)})
            for f in ('monthly','seasonal','annual'):
                temps.append(temporal_metrics(r,f).assign(controller=controller.name,tariff=tariff,allocation_method='proportional'))
                dis.append(disaggregated_metrics(r,env.ids,f).assign(controller=controller.name,tariff=tariff,allocation_method='proportional'))
            pros.append(prosumer_metrics(r,env.ids).assign(controller=controller.name,tariff=tariff,allocation_method='proportional'))
    temporal=pd.concat(temps,ignore_index=True); pd.concat(pros,ignore_index=True).to_csv(out/'tables/prosumer_metrics_revision.csv',index=False); pd.concat(dis,ignore_index=True).to_csv(out/'tables/disaggregated_prosumer_metrics_revision.csv',index=False)
    temporal.to_csv(out/'tables/temporal_metrics_revision.csv',index=False); pd.DataFrame(rows).to_csv(out/'tables/controller_tariff_summary_revision.csv',index=False); pd.DataFrame(checks).to_csv(out/'tables/revision_validation_checks.csv',index=False)
    m=temporal[(temporal.frequency=='monthly')&(temporal.tariff=='tou')]
    s=temporal[(temporal.frequency=='seasonal')&(temporal.tariff=='tou')]; s=s[s.period.str.endswith(('Summer','Winter'))]
    a=temporal[(temporal.frequency=='annual')&(temporal.tariff=='tou')]
    plot_controller_comparison(m,out/'figures','Monthly','Fig9a_monthly_controller_boxplots.png')
    plot_controller_comparison(s,out/'figures','Seasonal (Summer/Winter)','Fig9b_seasonal_summer_winter_controller_boxplots.png')
    plot_controller_comparison(a,out/'figures','Annual','Fig9c_annual_controller_boxplots.png')
    print('Revision 01 complete:',out.resolve())

if __name__=='__main__': main()
