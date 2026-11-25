"""Experiment 45: can SST see the observed subtidal mooring flow? Daily-mean mooring u,v x box-fitted daily SST gradient (7x7 px) vs box-mean daily dT/dt. Result: large-scale gradient ~0.5 K/100 km, -u.gradT rms 0.03-0.05 K/day vs dT/dt rms 0.11 K/day, correlation 0.04 pooled: the slow flow is below what SST resolves at this scale."""
import numpy as np, pandas as pd, xarray as xr, warnings; warnings.filterwarnings("ignore")
from scipy.ndimage import uniform_filter
B='/mnt/c/Users/00099894/Large_Data/Other/browseSST/'
c=xr.open_dataset(B+'test_bb23_2023.nc'); T=c.sst.values.astype(float); lon,lat=c.lon.values,c.lat.values; t=c.time.values
dxk=np.diff(lon).mean()*111.32e3*np.cos(np.deg2rad(14)); dyk=np.diff(lat).mean()*111.2e3
nd=T.shape[0]//24; Td=np.nanmean(T[:nd*24].reshape(nd,24,*T.shape[1:]),axis=1)          # daily mean SST (K)
print('daily SST field: finite fraction',np.isfinite(Td).mean().round(2))
d=xr.load_dataset('/mnt/c/Users/00099894/Large_Data/Papers/drifterOI_paper2/observations/BB23_Obs_vector_150m_long.nc')
df=pd.DataFrame({'time':d.time.values,'lon':d.longitude.values,'lat':d.latitude.values,'u':d.east_vel.values,'v':d.north_vel.values,'p':d.dtype.values.astype(str)})
mo=df[(df.p=='mADCP')&(df.time>='2023-04-01')&(df.time<'2023-06-17')].dropna(subset=['u','v']).copy(); mo['site']=mo.groupby([mo.lon.round(2),mo.lat.round(2)]).ngroup(); mo['day']=mo.time.dt.floor('1D')
days=pd.DatetimeIndex(t[:nd*24:24])
print('site  lon lat  ndays | |grad T| K/100km | corr(-u.gradT, dTdt)  slope | corr(-u_perp..) | rms obs u')
allx=[];ally=[]
for s,g in mo.groupby('site'):
    lo,la=g.lon.mean(),g.lat.mean(); j=np.abs(lat-la).argmin(); i=np.abs(lon-lo).argmin()
    # smooth box of 5x5 px around the site on daily fields: gradient from the box mean, dT/dt from the box-mean time series
    sl=(slice(max(j-3,0),j+4),slice(max(i-3,0),i+4)); box=Td[:,sl[0],sl[1]]
    Tm=np.nanmean(box,axis=(1,2)); 
    gy,gx=[],[]
    for k in range(nd):
        b=box[k]; 
        if np.isfinite(b).mean()<0.6: gx.append(np.nan); gy.append(np.nan); continue
        yy,xx=np.mgrid[:b.shape[0],:b.shape[1]]; m=np.isfinite(b); A=np.c_[xx[m]*dxk,yy[m]*dyk,np.ones(m.sum())]; co=np.linalg.lstsq(A,b[m],rcond=None)[0]; gx.append(co[0]); gy.append(co[1])
    gx,gy=np.array(gx),np.array(gy)
    ud=g.groupby('day').u.mean().reindex(days).values; vd=g.groupby('day').v.mean().reindex(days).values
    adv=-(ud*gx+vd*gy)*86400                       # K/day
    dTdt=np.gradient(Tm)                            # K/day (box-mean)
    m=np.isfinite(adv)&np.isfinite(dTdt)
    if m.sum()<15: continue
    cc=np.corrcoef(adv[m],dTdt[m])[0,1]; sl_=np.polyfit(adv[m],dTdt[m],1)[0]
    gm=np.hypot(gx,gy)[m].mean()*1e5
    print(f"  {s}  {lo:.2f} {la:.2f}  {m.sum():3d}  |  {gm:5.2f}  |  corr {cc:5.2f}  slope {sl_:5.2f}  | adv rms {adv[m].std():.3f} K/d  dTdt rms {dTdt[m].std():.3f} K/d | obs u rms {np.sqrt(np.nanmean(ud**2+vd**2)):.2f}")
    allx.append(adv[m]);ally.append(dTdt[m])
print('pooled corr',np.corrcoef(np.concatenate(allx),np.concatenate(ally))[0,1].round(2))
