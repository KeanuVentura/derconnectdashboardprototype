"""Reproducible 15-minute, explicitly fictional campus-meter investigation fixture."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'public'/'data'
SEED=67
STEP_H=0.25
START=pd.Timestamp('2026-07-06T00:00:00Z')
N_DAYS=70
TRAIN_DAYS=56
METERS=[
  ('B-01','M-101','AHU','Academic Hall',33,0.9),
  ('B-01','M-102','Lighting','Academic Hall',12,0.1),
  ('B-02','M-201','AHU','Research Center',40,1.3),
  ('B-02','M-202','Plug','Research Center',18,0.15),
  ('B-03','M-301','AHU','Library Annex',29,0.8),
  ('B-03','M-302','Lighting','Library Annex',15,0.1),
]
INJECTIONS=[
 ('E-101','M-101',59,9,18,'sustained_offset',14.0,'excess operation'),
 ('E-201','M-201',61,8,13,'ramp',16.0,'progressive load increase'),
 ('E-301','M-302',63,19,23,'sustained_offset',11.0,'after-hours lighting'),
 ('E-202','M-202',65,11,12,'pulse',15.0,'short load spike'),
]

def make(seed=SEED):
 rng=np.random.default_rng(seed)
 ts=pd.date_range(START, periods=N_DAYS*96, freq='15min', tz='UTC')
 hour=np.asarray(ts.hour)+np.asarray(ts.minute)/60
 day=np.asarray((ts-START).days)
 dow=np.asarray(ts.dayofweek)
 temp=19+6*np.sin(2*np.pi*(hour-8)/24)+2*np.sin(2*np.pi*day/12)+rng.normal(0,.35,len(ts))
 rows=[]; truths=[]
 for building,meter,category,name,capacity,weather_coef in METERS:
  occupied=((hour>=8)&(hour<18)&(dow<5)).astype(float)
  profile=({ 'AHU':.52,'Lighting':.22,'Plug':.58}[category]+{'AHU':.52,'Lighting':.78,'Plug':.35}[category]*occupied)
  scheduled=(((day==60)&(hour>=8)&(hour<18)) | ((day==66)&(hour>=8)&(hour<18))) if meter=='M-102' else np.zeros(len(ts),bool)
  schedule_multiplier=np.where(scheduled,1.58,1.0)
  baseline=capacity*profile*schedule_multiplier+weather_coef*np.maximum(temp-23,0)
  noise=rng.normal(0, max(0.5,capacity*.022),len(ts))
  measured=np.maximum(0,baseline+noise)
  for eid,mid,d,h1,h2,kind,amp,cause in INJECTIONS:
   if meter!=mid:continue
   mask=(day==d)&(hour>=h1)&(hour<h2)
   if kind=='ramp':delta=np.linspace(1.0,amp,int(mask.sum()))
   else:delta=np.full(mask.sum(),amp)
   measured[mask]+=delta
   truths.append(dict(event_id=eid,meter_id=meter,building_id=building,start=(START+pd.Timedelta(days=d,hours=h1)).isoformat(),end=(START+pd.Timedelta(days=d,hours=h2)).isoformat(),cause=cause,statistical_pattern='trend' if kind=='ramp' else 'collective shift' if kind=='sustained_offset' else 'point spike',amplitude_kw=amp))
  # Artificial missingness (including part of an injected event): retain rows and timestamps.
  missing=((day==62)&(hour>=2)&(hour<3)&(meter=='M-301')) | ((day==59)&(hour>=12)&(hour<12.5)&(meter=='M-101'))
  measured[missing]=np.nan
  rows.append(pd.DataFrame(dict(timestamp=ts,building_id=building,building_name=name,meter_id=meter,category=category,power_kw=measured,temperature_c=temp,occupied=occupied,scheduled_change=scheduled,quality=np.where(missing,'missing','ok'),interval_hours=STEP_H,clean_counterfactual_kw=baseline)))
 df=pd.concat(rows,ignore_index=True)
 return df,pd.DataFrame(truths)

def analyze(df):
 df=df.copy().sort_values(['meter_id','timestamp'])
 df['day']=((df.timestamp-START).dt.days).astype(int)
 df['slot']=df.timestamp.dt.dayofweek*96+df.timestamp.dt.hour*4+df.timestamp.dt.minute//15
 history=df[df.day<TRAIN_DAYS].copy()
 # Only past pre-evaluation observations define lookup; no injected events in history.
 history['adjusted_kw']=history.power_kw-0.3*np.maximum(history.temperature_c-23,0)
 lookup=history.groupby(['meter_id','slot','occupied'])['adjusted_kw'].agg(['median','count']).reset_index()
 df=df.merge(lookup,how='left',on=['meter_id','slot','occupied'])
 fallback=history.groupby(['meter_id','occupied'])['adjusted_kw'].median().rename('fallback_kw').reset_index()
 df=df.merge(fallback,on=['meter_id','occupied'],how='left')
 df['history_n']=df['count'].fillna(0).astype(int)
 df['expected_kw']=np.where(df.history_n>=3,df['median'],df['fallback_kw'])+0.3*np.maximum(df.temperature_c-23,0)
 # Synthetic schedule event is explicitly logged and corrected; do not treat legitimate operation as fault.
 df['expected_kw']*=np.where(df.scheduled_change,1.58,1)
 # Stable meter-specific residual scale from training; conservative floor prevents overfitting.
 tr=df[df.day<TRAIN_DAYS]
 scales=tr.assign(resid=lambda x: x.power_kw-x.expected_kw).groupby('meter_id')['resid'].apply(lambda x:max(1.5,1.4826*np.nanmedian(np.abs(x-np.nanmedian(x))))).rename('scale')
 df=df.join(scales,on='meter_id')
 df['residual_kw']=df.power_kw-df.expected_kw
 df['is_flag']=((df.residual_kw>np.maximum(3.5*df.scale,0.22*df.expected_kw)) & df.quality.eq('ok') & (df.day>=TRAIN_DAYS)).fillna(False)
 df['positive_excess_kwh']=np.maximum(df.residual_kw,0).fillna(0)*df.interval_hours
 # Prioritize contiguous runs of >=2 flag samples as alerts; distinguish alerts from injected truth.
 detected=[]
 for meter,g in df[df.day>=TRAIN_DAYS].groupby('meter_id',sort=True):
  g=g.sort_values('timestamp'); vals=g.is_flag.to_numpy(); inds=np.flatnonzero(np.diff(np.r_[False,vals,False]));
  for left,right in zip(inds[::2],inds[1::2]):
   if right-left<2: continue
   z=g.iloc[left:right]
   detected.append(dict(event_id=f'A-{len(detected)+1:03}',meter_id=meter,building_id=z.building_id.iloc[0],building_name=z.building_name.iloc[0],category=z.category.iloc[0],start=z.timestamp.iloc[0].isoformat(),end=(z.timestamp.iloc[-1]+pd.Timedelta(minutes=15)).isoformat(),duration_hours=round(len(z)*STEP_H,2),excess_kwh=round(z.positive_excess_kwh.sum(),2),peak_deviation_kw=round(z.residual_kw.max(),2),mean_deviation_pct=round(100*(z.residual_kw/z.expected_kw).mean(),1),quality_pct=round(100*z.quality.eq('ok').mean(),1),priority_score=round(min(100,8*np.log1p(z.positive_excess_kwh.sum())+4*np.log1p(len(z)*STEP_H)+2*np.log1p(z.residual_kw.max())),1)))
 alerts=pd.DataFrame(detected)
 return df,alerts

def evaluate(df,truth,alerts):
 # One-to-one overlap matching, with separate event TP/FP/FN and first-alert latency.
 remaining=set(truth.event_id); matched=[]; false=[]
 for a in alerts.to_dict('records'):
  candidates=[]
  for t in truth.to_dict('records'):
   if t['event_id'] not in remaining or t['meter_id']!=a['meter_id']:continue
   overlap=min(pd.Timestamp(t['end']),pd.Timestamp(a['end']))-max(pd.Timestamp(t['start']),pd.Timestamp(a['start']))
   if overlap.total_seconds()>0:candidates.append((overlap.total_seconds(),t))
  if candidates:
   t=max(candidates,key=lambda x:x[0])[1];remaining.remove(t['event_id']);matched.append({'true_event':t['event_id'],'alert_event':a['event_id'],'latency_minutes':round((pd.Timestamp(a['start'])-pd.Timestamp(t['start'])).total_seconds()/60,1)})
  else:false.append(a['event_id'])
 tp=len(matched);fp=len(false);fn=len(remaining)
 return {'event_precision':round(tp/(tp+fp),3) if tp+fp else None,'event_recall':round(tp/(tp+fn),3) if tp+fn else None,'true_positive_events':tp,'false_positive_events':fp,'missed_events':fn,'matches':matched,'unmatched_alerts':false,'missed_truth_events':sorted(remaining),'scope':'Synthetic holdout only; not real-world accuracy'}

def save():
 df,truth=make(); analyzed,alerts=analyze(df); ev=evaluate(analyzed,truth,alerts)
 OUT.mkdir(parents=True,exist_ok=True)
 view=analyzed[analyzed.day>=TRAIN_DAYS].copy()
 view['timestamp']=view.timestamp.dt.strftime('%Y-%m-%dT%H:%M:%SZ')
 fields=['timestamp','building_id','building_name','meter_id','category','power_kw','expected_kw','temperature_c','occupied','scheduled_change','quality','interval_hours','residual_kw','positive_excess_kwh','history_n']
 view[fields].round(3).to_csv(OUT/'observations.csv',index=False)
 alerts.sort_values('priority_score',ascending=False).to_json(OUT/'alerts.json',orient='records',indent=2)
 truth.to_json(OUT/'ground_truth.json',orient='records',indent=2)
 (OUT/'evaluation.json').write_text(json.dumps(ev,indent=2))
 (OUT/'metadata.json').write_text(json.dumps({'dataset':'SYNTHETIC DEMONSTRATION ONLY','seed':SEED,'step_minutes':15,'buildings':3,'meters':6,'training_days':TRAIN_DAYS,'evaluation_days':N_DAYS-TRAIN_DAYS,'weather_adjustment_kw_per_c':0.3},indent=2))
 print(json.dumps({'samples':len(df),'holdout_samples':len(view),'alerts':len(alerts),**{k:v for k,v in ev.items() if k in ('event_precision','event_recall','true_positive_events','false_positive_events','missed_events')}},indent=2))
 return df,truth,analyzed,alerts,ev

if __name__=='__main__':save()
