import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from pipeline import make,analyze,evaluate,STEP_H

def test_reproducible():
 a,_=make(67);b,_=make(67)
 assert a.power_kw.equals(b.power_kw)

def test_interval_and_nonnegative():
 df,_=make();assert (df.interval_hours==STEP_H).all()
 assert df.timestamp.notna().all() and (df.power_kw.dropna()>=0).all()
 assert df.groupby('meter_id').timestamp.apply(lambda x:x.is_monotonic_increasing).all()

def test_energy_integral():
 df,_=make(); d,alerts=analyze(df)
 assert np.allclose(d.positive_excess_kwh,np.maximum(d.residual_kw,0).fillna(0)*STEP_H)
 assert all(alerts.excess_kwh>=0)

def test_aggregation():
 df,_=make(); g=df.groupby(['timestamp','building_id']).power_kw.sum(min_count=1)
 manual=df.query("building_id=='B-01'").pivot(index='timestamp',columns='meter_id',values='power_kw').sum(axis=1,min_count=1)
 assert np.allclose(g.xs('B-01',level='building_id'),manual,equal_nan=True)

def test_holdout_and_baseline():
 df,t=make();d,a=analyze(df)
 assert d.loc[d.day<56,'is_flag'].sum()==0
 assert d.loc[d.day>=56,'expected_kw'].notna().all()
 assert d.loc[d.quality=='missing','is_flag'].sum()==0
 assert not a.empty
 assert evaluate(d,t,a)['event_recall']>=0

def test_schedule_legitimate():
 df,_=make();d,_=analyze(df)
 changed=d[d.scheduled_change]
 assert not changed.empty
 assert changed.is_flag.sum()==0
