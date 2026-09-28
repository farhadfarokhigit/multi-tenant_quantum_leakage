"""The frame hierarchy measured on an active victim."""
import csv, math
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

COL={'bare':'#B03A2E','local':'#B7950B','full':'#1F4E79'}
LAB={'bare':'no frame','local':'boundary-only frame','full':'full-register frame'}
MRK={'bare':'o','local':'s','full':'^'}
SH={'bare':4096,'local':4096*4,'full':4096*16}
ETA=0.01
R={}
for r in csv.DictReader(open('hierarchy_data.csv')):
    R.setdefault(r['arm'],[]).append({k:(float(v) if k!='arm' else v) for k,v in r.items()})

fig,ax=plt.subplots(1,3,figsize=(11.0,3.3))
tg=np.linspace(0.1,118,200)
for arm in ('bare','local','full'):
    rows=R[arm]
    t=np.array([r['delay_us'] for r in rows]); y=np.array([r['dtheta'] for r in rows])
    g=np.array([r['gamma'] for r in rows]); c=np.array([r['contrast'] for r in rows])
    s=math.sqrt(2)/math.sqrt(SH[arm])/c
    sigma_c=(1.0/math.sqrt(SH[arm]))/math.sqrt(2.0)
    sigma_g=np.sqrt((np.abs(np.sin(y/2))*sigma_c)**2 +
                     (0.5*c*np.abs(np.cos(y/2))*s)**2)

    # weighted fit on every point, no contrast cut
    w=1/s**2; S,Sx,Sy=w.sum(),(w*t).sum(),(w*y).sum()
    Sxx,Sxy=(w*t*t).sum(),(w*t*y).sum(); d=S*Sxx-Sx**2
    a=(S*Sxy-Sx*Sy)/d; b=(Sxx*Sy-Sx*Sxy)/d
    phase_rate=a  # rad/us, the fitted slope of dtheta against tau

    # contrast decay fit, also on every point
    slope_c,inter_c=np.polyfit(t,np.log(c),1)
    env=np.exp(inter_c)*np.exp(slope_c*tg)*np.abs(np.sin(phase_rate*tg/2))

    mf='none' if arm=='local' else COL[arm]
    ms=6.5 if arm=='local' else 4
    ax[0].errorbar(t,y,yerr=s,fmt=MRK[arm],ms=ms,color=COL[arm],mfc=mf,
                   label=LAB[arm],capsize=2,lw=1,mew=1.1)
    ax[0].plot(tg,b+a*tg,'-',color=COL[arm],lw=1,alpha=.85)
    ax[1].errorbar(t,g,yerr=sigma_g,fmt=MRK[arm],ms=ms,color=COL[arm],
                   mfc=mf,mew=1.1,capsize=2,elinewidth=0.7,label=LAB[arm],ls='none')
    ax[1].plot(tg,env,'-',color=COL[arm],lw=1,alpha=.85)

    # (c) shot budget, same construction as make_figure.py: a theoretical
    # line from the fitted envelope, and markers computed directly from
    # each measured Gamma, propagated in log space since N* is steeply
    # nonlinear in Gamma
    with np.errstate(divide='ignore'):
        nstar_theory=2.0*math.log(1/ETA)/env**2
    nstar_meas=2.0*math.log(1/ETA)/g**2
    ax[2].plot(tg,nstar_theory,'-',color=COL[arm],lw=1,alpha=.85)
    ax[2].plot(t,nstar_meas,MRK[arm],ms=ms,color=COL[arm],mfc=mf,mew=1.1,
               label=LAB[arm],ls='none')

ax[0].set_xlabel(r'idle window $\tau$ ($\mu$s)');
ax[0].set_ylabel(r'phase gap $\Delta\theta$ (rad)')
ax[0].legend(fontsize=6.5,frameon=False,loc='upper left');
ax[0].set_title('a',loc='left',fontsize=9,fontweight='bold')
ax[1].set_yscale('log');
ax[1].set_ylim(1e-3,3.0)
ax[1].set_xlabel(r'idle window $\tau$ ($\mu$s)');
ax[1].set_ylabel(r'conditional spread $\Gamma$')
ax[1].legend(fontsize=6.5,frameon=False,loc='center left',bbox_to_anchor=(0.02,0.42));
ax[1].set_title('b',loc='left',fontsize=9,fontweight='bold')
ax[2].set_yscale('log')
ax[2].set_xlabel(r'idle window $\tau$ ($\mu$s)');
ax[2].set_ylabel(r'shot budget $N^{*}$')
ax[2].legend(fontsize=6.5,frameon=False,loc='upper left');
ax[2].set_title('c',loc='left',fontsize=9,fontweight='bold')
for a_ in ax:
    a_.tick_params(labelsize=8);
    a_.set_xlim(0,118); 
    a_.grid(True,alpha=0.3,lw=0.5)
fig.tight_layout(); fig.savefig('hierarchy.pdf');
fig.savefig('hierarchy.png',dpi=200)
print('built')
