from __future__ import annotations
import argparse
import pandas as pd
from microgrid_opt.data import load_forecast_csv

p = argparse.ArgumentParser(description='Validate client long-format data before a final run.')
p.add_argument('csv')
a = p.parse_args()
df = load_forecast_csv(a.csv)
print('VALID')
print('rows:', len(df))
print('hours:', df['timestamp'].nunique())
print('prosumers:', df['prosumer_id'].nunique())
print('date range:', df['timestamp'].min(), 'to', df['timestamp'].max())
print('forecast columns explicit:', {'load_forecast_kwh','pv_forecast_kwh'}.issubset(pd.read_csv(a.csv,nrows=1).columns))
print('RTP available:', df['rtp_cad_per_kwh'].notna().all())
