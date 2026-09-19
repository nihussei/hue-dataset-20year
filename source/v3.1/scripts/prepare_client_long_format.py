from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

from microgrid_opt.data import validate_long_data


def _timestamp(df: pd.DataFrame) -> pd.Series:
    if 'timestamp' in df:
        return pd.to_datetime(df['timestamp'])
    if 'datetime' in df:
        return pd.to_datetime(df['datetime'])
    if 'date' in df and 'hour' in df:
        hour = pd.to_numeric(df['hour'], errors='raise')
        # Client source code generally uses 1-indexed hour after cleanup.
        offset = np.where(hour >= 1, hour - 1, hour)
        return pd.to_datetime(df['date']) + pd.to_timedelta(offset, unit='h')
    raise ValueError('Could not infer timestamp: need timestamp, datetime, or date+hour')


def _pid_from_filename(path: str) -> str:
    m = re.search(r'(?:P|Residential_?)(\d+)', Path(path).stem, re.I)
    if not m:
        raise ValueError(f'Could not infer prosumer ID from {path}')
    return f'P{int(m.group(1))}'


def load_load_forecasts(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f'No load files matched: {pattern}')
    rows = []
    for path in files:
        d = pd.read_csv(path)
        pid = _pid_from_filename(path)
        col = 'energy_kWh' if 'energy_kWh' in d else ('load_kwh' if 'load_kwh' in d else None)
        if col is None:
            raise ValueError(f'{path}: expected energy_kWh or load_kwh')
        rows.append(pd.DataFrame({
            'timestamp': _timestamp(d),
            'prosumer_id': pid,
            'load_forecast_kwh': pd.to_numeric(d[col], errors='raise').clip(lower=0),
        }))
    return pd.concat(rows, ignore_index=True)


def load_pv(path: str, prosumer_ids: list[str]) -> pd.DataFrame:
    d = pd.read_csv(path)
    ts = _timestamp(d)
    if {'prosumer_id', 'pv_kwh'}.issubset(d.columns):
        return pd.DataFrame({'timestamp': ts, 'prosumer_id': d['prosumer_id'].astype(str),
                             'pv_forecast_kwh': pd.to_numeric(d['pv_kwh'], errors='raise').clip(lower=0)})
    col = 'ac_output' if 'ac_output' in d else ('pv_kwh' if 'pv_kwh' in d else None)
    if col is None:
        raise ValueError('PV file needs long-format prosumer_id+pv_kwh or a common ac_output/pv_kwh series')
    # A common PV series is repeated only when the user explicitly supplies such
    # a file. Household-specific scaling should be applied upstream if required.
    common = pd.DataFrame({'timestamp': ts, 'pv_forecast_kwh': pd.to_numeric(d[col], errors='raise').clip(lower=0)})
    return pd.concat([common.assign(prosumer_id=pid) for pid in prosumer_ids], ignore_index=True)


def main():
    p = argparse.ArgumentParser(description='Assemble client forecast outputs into the milestone long format.')
    p.add_argument('--load-pattern', required=True, help=r'Glob such as C:\forecasts\df_P*_20yr_full.csv')
    p.add_argument('--pv-csv', required=True, help='PV forecast CSV')
    p.add_argument('--realized-csv', help='Optional long-format realized timestamp/prosumer_id/load_kwh/pv_kwh data')
    p.add_argument('--rtp-csv', help='Optional timestamp,rtp_cad_per_kwh[,rtp_forecast_cad_per_kwh] CSV')
    p.add_argument('--output', default='client_long_format.csv')
    a = p.parse_args()

    load = load_load_forecasts(a.load_pattern)
    pids = sorted(load['prosumer_id'].unique())
    pv = load_pv(a.pv_csv, pids)
    out = load.merge(pv, on=['timestamp', 'prosumer_id'], how='inner')

    if a.realized_csv:
        realized = pd.read_csv(a.realized_csv)
        realized['timestamp'] = pd.to_datetime(realized['timestamp'])
        needed = ['timestamp','prosumer_id','load_kwh','pv_kwh']
        out = out.merge(realized[needed], on=['timestamp','prosumer_id'], how='left')
    else:
        # Forecast-only compatibility: use forecasts as realized placeholders.
        # Final forecast-error studies should supply actual realized columns.
        out['load_kwh'] = out['load_forecast_kwh']
        out['pv_kwh'] = out['pv_forecast_kwh']

    if a.rtp_csv:
        rtp = pd.read_csv(a.rtp_csv)
        rtp['timestamp'] = pd.to_datetime(rtp['timestamp'])
        cols = ['timestamp','rtp_cad_per_kwh']
        if 'rtp_forecast_cad_per_kwh' in rtp: cols.append('rtp_forecast_cad_per_kwh')
        out = out.merge(rtp[cols], on='timestamp', how='left')
    out['critical_fraction'] = 1.0
    out['grid_available'] = 1
    out = validate_long_data(out)
    out.to_csv(a.output, index=False)
    print(f'Wrote {a.output}: {out.prosumer_id.nunique()} prosumers, {out.timestamp.nunique()} hours')


if __name__ == '__main__':
    main()
