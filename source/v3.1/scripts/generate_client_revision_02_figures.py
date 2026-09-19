#!/usr/bin/env python3
"""Generate Client Revision 02 figure extensions from validated result tables.

Outputs
-------
- 28 Figure 9 PNGs: 7 views x 4 tariffs (Flat, Tiered, ToU, RTP)
- 4 Figure 12 PNGs: allocation-method comparison for each tariff

The script is deterministic and uses the existing validated held-out 2040 result
products. No controller retraining is performed.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CONTROLLERS = ["rule_based", "mpc", "d3qn_per", "d3qn_uniform_replay"]
CTRL_LABELS = {
    "rule_based": "Rule-Based",
    "mpc": "MPC",
    "d3qn_per": "D3QN-PER",
    "d3qn_uniform_replay": "D3QN-Uniform",
}
TARIFFS = ["flat", "tiered", "tou", "rtp"]
TARIFF_LABELS = {"flat": "Flat", "tiered": "Tiered", "tou": "ToU", "rtp": "RTP"}
METRICS = [
    ("net_cost_cad", "Net cost (CAD)"),
    ("pv_utilization", "PV utilization"),
    ("peak_load_reduction_kw", "Peak-load reduction (kW)"),
    ("cumulative_reward", "Cumulative reward"),
]
PROSUMER_METRICS = [
    ("net_cost_cad", "Net cost (CAD)"),
    ("pv_utilization", "PV utilization"),
    ("peak_load_reduction_kw", "Peak-load reduction (kW)"),
    ("cumulative_utility", "Prosumer utility"),
]


def _style_axis(ax):
    ax.grid(axis="y", alpha=0.22)
    ax.set_axisbelow(True)


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def aggregate_monthly(temporal, tariff, out):
    d = temporal[(temporal.tariff == tariff) & (temporal.frequency == "monthly")]
    fig, axs = plt.subplots(2, 2, figsize=(13, 8.5))
    for ax, (metric, title) in zip(axs.flat, METRICS):
        vals = [d[d.controller == c][metric].dropna().values for c in CONTROLLERS]
        ax.boxplot(vals, labels=[CTRL_LABELS[c] for c in CONTROLLERS], patch_artist=True)
        ax.set_title(title, fontweight="bold")
        _style_axis(ax)
    fig.suptitle(f"Figure 9 - {TARIFF_LABELS[tariff]}: Monthly Controller Box Plots (12 months)", fontsize=16, fontweight="bold")
    _save(fig, out / f"fig9_{tariff}_01_monthly_aggregate.png")


def aggregate_seasonal(temporal, tariff, out):
    # Client requested Summer and Winter. Use the three monthly observations in each season.
    d = temporal[(temporal.tariff == tariff) & (temporal.frequency == "monthly")].copy()
    d["month"] = pd.to_datetime(d.period + "-01").dt.month
    seasons = {"Summer": [6, 7, 8], "Winter": [12, 1, 2]}
    fig, axs = plt.subplots(2, 2, figsize=(13, 8.5))
    for ax, (metric, title) in zip(axs.flat, METRICS):
        positions=[]; data=[]; labels=[]
        x=1.0
        for c in CONTROLLERS:
            for s, months in seasons.items():
                data.append(d[(d.controller==c) & d.month.isin(months)][metric].dropna().values)
                positions.append(x)
                labels.append(f"{CTRL_LABELS[c]}\n{s}")
                x += 1.0
            x += 0.35
        bp=ax.boxplot(data, positions=positions, widths=0.68, patch_artist=True)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_title(title, fontweight="bold")
        _style_axis(ax)
    fig.suptitle(f"Figure 9 - {TARIFF_LABELS[tariff]}: Summer/Winter Controller Box Plots", fontsize=16, fontweight="bold")
    _save(fig, out / f"fig9_{tariff}_02_summer_winter_aggregate.png")


def aggregate_annual(pros, util, tariff, out):
    # Annual aggregate distributions use the ten prosumers as the observations.
    p = pros[(pros.tariff == tariff) & (pros.allocation_method == "proportional")]
    u = util[(util.tariff == tariff) & (util.frequency == "annual")]
    fig, axs = plt.subplots(2, 2, figsize=(13, 8.5))
    for ax, (metric, title) in zip(axs.flat, METRICS):
        if metric == "cumulative_reward":
            vals=[u[u.controller==c].cumulative_utility.dropna().values for c in CONTROLLERS]
        else:
            vals=[p[p.controller==c][metric].dropna().values for c in CONTROLLERS]
        ax.boxplot(vals, labels=[CTRL_LABELS[c] for c in CONTROLLERS], patch_artist=True)
        ax.set_title(title, fontweight="bold")
        _style_axis(ax)
    fig.suptitle(f"Figure 9 - {TARIFF_LABELS[tariff]}: Annual Prosumer Distribution (10 prosumers)", fontsize=16, fontweight="bold")
    _save(fig, out / f"fig9_{tariff}_03_annual_aggregate.png")


def _per_prosumer_box(source, tariff, frequency, period_filter, metric, ax, annual=False):
    d=source[(source.tariff==tariff) & (source.frequency==frequency)].copy()
    if period_filter is not None:
        d=d[d.period.apply(period_filter)]
    prosumers=sorted(d.prosumer_id.unique(), key=lambda x:int(str(x).replace('P','')))
    base=np.arange(len(prosumers))*5.5 + 1.0
    offsets=[-1.45,-0.48,0.48,1.45]
    width=0.82
    for ci,c in enumerate(CONTROLLERS):
        vals=[]; pos=[]
        for i,pid in enumerate(prosumers):
            a=d[(d.controller==c)&(d.prosumer_id==pid)][metric].dropna().values
            vals.append(a)
            pos.append(base[i]+offsets[ci])
        if annual:
            for a,x in zip(vals,pos):
                if len(a): ax.plot(x, a[0], 'o', markersize=4)
        else:
            ax.boxplot(vals, positions=pos, widths=width, patch_artist=True, manage_ticks=False)
    ax.set_xticks(base)
    ax.set_xticklabels(prosumers, rotation=0, fontsize=8)
    _style_axis(ax)


def per_prosumer_view(source, tariff, kind, out):
    if kind == "monthly":
        frequency="monthly"; filt=None; title="Monthly per-prosumer box plots (12 observations per box)"; annual=False
    elif kind == "summer":
        frequency="monthly"; filt=lambda p:int(str(p)[5:7]) in [6,7,8]; title="Summer per-prosumer box plots (Jun-Aug)"; annual=False
    elif kind == "winter":
        frequency="monthly"; filt=lambda p:int(str(p)[5:7]) in [12,1,2]; title="Winter per-prosumer box plots (Dec-Feb)"; annual=False
    elif kind == "annual":
        frequency="annual"; filt=None; title="Annual per-prosumer values"; annual=True
    else:
        raise ValueError(kind)
    fig, axs=plt.subplots(2,2,figsize=(15,9))
    for ax,(metric,label) in zip(axs.flat,PROSUMER_METRICS):
        _per_prosumer_box(source,tariff,frequency,filt,metric,ax,annual=annual)
        ax.set_title(label,fontweight='bold')
    handles=[plt.Line2D([0],[0],marker='s',linestyle='None',label=CTRL_LABELS[c]) for c in CONTROLLERS]
    fig.legend(handles=handles,loc='upper center',ncol=4,bbox_to_anchor=(0.5,0.935))
    fig.suptitle(f"Figure 9 - {TARIFF_LABELS[tariff]}: {title}",fontsize=16,fontweight='bold',y=0.985)
    idx={"monthly":"04","summer":"05","winter":"06","annual":"07"}[kind]
    _save(fig,out/f"fig9_{tariff}_{idx}_{kind}_per_prosumer.png")


def figure12(allocation, tariff, out):
    d=allocation[allocation.tariff.astype(str)==tariff].copy()
    metrics=[
        ("cumulative_reward","Cumulative reward"),
        ("net_cost_cad","Net cost (CAD)"),
        ("pv_utilization","PV utilization"),
        ("CAIFI_interruptions_affected_customer","CAIFI (interruptions/affected customer)"),
    ]
    allocs=['proportional','priority','contribution']
    fig,axs=plt.subplots(2,2,figsize=(13.5,8.5))
    x=np.arange(len(allocs)); width=0.18
    for ax,(metric,title) in zip(axs.flat,metrics):
        for i,c in enumerate(CONTROLLERS):
            vals=[]
            for a in allocs:
                q=d[(d.controller.astype(str)==c)&(d.allocation_method.astype(str)==a)]
                vals.append(float(q.iloc[0][metric]) if len(q) else np.nan)
            ax.bar(x+(i-1.5)*width,vals,width,label=CTRL_LABELS[c])
        ax.set_xticks(x); ax.set_xticklabels(['Proportional','Priority','Contribution'])
        ax.set_title(title,fontweight='bold'); _style_axis(ax)
    axs[0,0].legend(fontsize=8,ncol=2)
    fig.suptitle(f"Figure 12 - {TARIFF_LABELS[tariff]}: Allocation Method Comparison",fontsize=16,fontweight='bold')
    _save(fig,out/f"fig12_{tariff}_allocation_method_comparison.png")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--results-dir',type=Path,required=True)
    args=ap.parse_args()
    tables=args.results_dir/'tables'; fig9=args.results_dir/'figures'/'revision02_figure9'; fig12=args.results_dir/'figures'/'revision02_figure12'
    temporal=pd.read_csv(tables/'figure9_aggregate_source.csv')
    per=pd.read_csv(tables/'figure9_per_prosumer_source.csv')
    pros=pd.read_csv(tables/'prosumer_metrics.csv')
    util=pd.read_csv(tables/'prosumer_utility_temporal.csv')
    alloc=pd.read_csv(tables/'allocation_method_comparison_all_tariffs.csv')
    for t in TARIFFS:
        aggregate_monthly(temporal,t,fig9)
        aggregate_seasonal(temporal,t,fig9)
        aggregate_annual(pros,util,t,fig9)
        for k in ['monthly','summer','winter','annual']:
            per_prosumer_view(per,t,k,fig9)
        figure12(alloc,t,fig12)
    print(f"Generated {len(list(fig9.glob('*.png')))} Figure 9 files and {len(list(fig12.glob('*.png')))} Figure 12 files")

if __name__=='__main__': main()
