"""Aggregate paired realizations and plot expansion relative to each grid minimum."""
from __future__ import annotations
from collections import defaultdict
import itertools
import math
import numpy as np
import _sweep_engine as engine

CELL_FIELDS=["image","a","noise_level","sigma_dn","phi","psi","phi_normalized","psi_normalized",
             "mean_stream_bits","mean_bits_per_sample","std_bits_per_sample","expansion_percent",
             "completed_repeats","expected_repeats","cell_complete","panel_complete",
             "max_reconstruction_error","maximum_clipped_fraction","nu_max_reached"]
BEST_FIELDS=["image","a","noise_level","sigma_dn","sigma_over_a","phi","psi","phi_normalized",
             "psi_normalized","best_mean_bits_per_sample","completed_cells","expected_cells",
             "complete","full_legal_grid","tied_best_pairs","nu_min","nu_max","t_inc","theta","first_band"]


def aggregate(records,study):
    groups=defaultdict(list);input_identity={}
    for record in records:
        r=record["row"]
        key=(r["image"],int(r["a"]),float(r["noise_level"]),int(r["phi"]),int(r["psi"]))
        groups[key].append(r)
        # The same image/noise realization must be used for every tested pair
        # and for different A values whenever their actual sigma is the same.
        nk=(r["image"],float(r["sigma_dn"]),int(r["noise_rep"]))
        ident=(r["input_sha256"],int(r["noise_seed"]),int(r["sample_count"]))
        if nk in input_identity and input_identity[nk]!=ident:
            raise ValueError("Different noisy inputs were used within a paired comparison.")
        input_identity[nk]=ident
    panels={};cells=[];best=[]
    L=2**study["profile"]["theta"]
    expected_cells=len(study["phi_values"])*len(study["psi_values"])
    for scene,A,level in itertools.product(study["inputs"],study["a_values"],study["noise_values"]):
        sigma=float(level*A if study["noise_mode"]=="ratio" else level)
        nrep=1 if sigma==0 else study["repeats"]
        selected=[]
        for phi,psi in itertools.product(study["phi_values"],study["psi_values"]):
            rr=groups.get((scene,A,float(level),phi,psi),[])
            ids=[int(r["noise_rep"]) for r in rr]
            if len(ids)!=len(set(ids)):raise ValueError("Duplicate realization in a grid cell.")
            complete=set(ids)==set(range(nrep))
            row=dict(image=scene,a=A,noise_level=level,sigma_dn=sigma,phi=phi,psi=psi,
                     phi_normalized=phi/L,psi_normalized=psi/L,mean_stream_bits="",mean_bits_per_sample="",
                     std_bits_per_sample="",expansion_percent="",completed_repeats=len(rr),expected_repeats=nrep,
                     cell_complete=int(complete),panel_complete=0,max_reconstruction_error="",
                     maximum_clipped_fraction="",nu_max_reached="")
            if complete:
                mean_bits=sum(int(r["stream_bits"]) for r in rr)/nrep
                row.update(mean_stream_bits=mean_bits,mean_bits_per_sample=mean_bits/int(rr[0]["sample_count"]),
                           std_bits_per_sample=float(np.std([float(r["bits_per_sample"]) for r in rr],ddof=1)) if nrep>1 else (0.0 if sigma==0 else ""),
                           max_reconstruction_error=max(int(r["max_error"]) for r in rr),
                           maximum_clipped_fraction=max(float(r["clipped_fraction"]) for r in rr),
                           nu_max_reached=int(all(int(r["nu_max_reached"]) for r in rr)))
            selected.append(row)
        usable=[r for r in selected if r["cell_complete"]]
        panel_complete=len(usable)==expected_cells
        if usable:
            minimum=min(r["mean_stream_bits"] for r in usable)
            ties=[r for r in usable if r["mean_stream_bits"]==minimum]
            winner=min(ties,key=lambda r:(r["phi"],r["psi"]))
            for r in usable:r["expansion_percent"]=max(0.0,100*(r["mean_stream_bits"]/minimum-1))
            entry={k:winner[k] for k in ("image","a","noise_level","sigma_dn","phi","psi","phi_normalized","psi_normalized")}
            entry.update(sigma_over_a=sigma/A,best_mean_bits_per_sample=winner["mean_bits_per_sample"],
                         completed_cells=len(usable),expected_cells=expected_cells,complete=int(panel_complete),
                         full_legal_grid=int(expected_cells==L*L),
                         tied_best_pairs=";".join(f"{r['phi']}:{r['psi']}" for r in ties))
            entry.update({k:study["profile"][k] for k in ("nu_min","nu_max","t_inc","theta","first_band")})
            best.append(entry)
        else:winner=None;ties=[]
        for r in selected:r["panel_complete"]=int(panel_complete)
        cells.extend(selected)
        panels[(scene,A,float(level))]=dict(cells=selected,best=winner,ties=ties,complete=panel_complete,
                                          completed=len(usable),expected=expected_cells,sigma=sigma)
    return panels,cells,best


def plot_results(records,metadata,out):
    if not records:return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap,Normalize
    study=metadata["study"];profile=study["profile"];L=2**profile["theta"]
    panels,cells,best=aggregate(records,study)
    engine.table(out/"heatmap_cells.csv",CELL_FIELDS,cells)
    engine.table(out/"best_pairs.csv",BEST_FIELDS,best)
    expansions=[float(r["expansion_percent"]) for r in cells if r["expansion_percent"]!=""]
    vmax=max(1.0,math.ceil(max(expansions,default=0)/10)*10)
    cmap=LinearSegmentedColormap.from_list("expansion",["#ffffff","#ffb0b0","#ff2525","#ffb000",
                                                      "#ffff00","#00c400","#00cfff","#0022cc","#000000"])
    cmap.set_bad("#b9bdc5")
    norm=Normalize(vmin=0,vmax=vmax)
    ticks=[0,.25,.5,.75];extent=(-.5/L,(L-.5)/L,-.5/L,(L-.5)/L)

    def draw_panel(ax,scene,A,level,show_image=False):
        panel=panels[(scene,A,float(level))]
        values=np.full((L,L),np.nan)
        for r in panel["cells"]:
            if r["cell_complete"]:values[r["phi"],r["psi"]]=r["expansion_percent"]
        im=ax.imshow(np.ma.masked_invalid(values),origin="lower",extent=extent,cmap=cmap,norm=norm,
                     interpolation="nearest",aspect="equal")
        title=(study['inputs'][scene]['label']+f" | A={A} DN\n") if show_image else ""
        title+=rf"Added $\sigma={panel['sigma']:g}$ DN ($\sigma/A={panel['sigma']/A:g}$)"
        win=panel["best"]
        if win:
            ax.scatter(win["psi"]/L,win["phi"]/L,marker="*",s=115,c="black",edgecolors="white",linewidths=.65)
            title+=f"\nBest tested: phi={win['phi']}, psi={win['psi']}; {win['mean_bits_per_sample']:.4f} bps"
            if len(panel["ties"])>1:title+=f" ({len(panel['ties'])} ties)"
        else:title+="\nNo complete cells yet"
        if not panel["complete"]:title+=f"\nPROVISIONAL: {panel['completed']}/{panel['expected']} cells"
        ax.set_title(title,fontsize=9)
        ax.set_xlabel(r"Normalized offset, $\psi'=\psi/2^\Theta$",fontsize=9)
        ax.set_ylabel(r"Normalized damping, $\phi'=\phi/2^\Theta$",fontsize=9)
        ax.set_xticks(ticks);ax.set_yticks(ticks)
        return im

    def finish(fig,axes,im,title,name):
        fig.suptitle(title,fontsize=12)
        fig.colorbar(im,ax=list(axes.flat),orientation="horizontal",fraction=.035,
                     pad=.16 if axes.shape[0]==1 else .10,
                     label="Increase over best tested compressed size (%)")
        fig.text(.5,.012,"Star: one best tested pair; all exact ties are listed in best_pairs.csv. Gray: untested/incomplete.\n"
                 "Each noise level has its own minimum; colors use one shared percentage scale. Sigma is ADDED noise.",
                 ha="center",fontsize=8)
        for suffix in ("png","pdf"):fig.savefig(out/(name+"."+suffix),dpi=160,bbox_inches="tight")
        plt.close(fig)

    settings=(f"P={profile['p']} | {profile['local_sum']}, {profile['prediction_mode']} | hybrid | "
              f"nu_min={profile['nu_min']}, FIXED nu_max={profile['nu_max']}, t_inc={profile['t_inc']}\n"
              f"Theta={profile['theta']} | first band={profile['first_band']} | "
              f"crop {study['crop'][0]} x {study['crop'][1]} | noise repeats={study['repeats']}")
    levels=study["noise_values"];cols=min(3,len(levels));rows=math.ceil(len(levels)/cols)
    for scene,A in itertools.product(study["inputs"],study["a_values"]):
        fig,axes=plt.subplots(rows,cols,squeeze=False,figsize=(max(8.0,4.7*cols),4.5*rows+2.3))
        fig.subplots_adjust(top=.79 if rows==1 else .85,bottom=.19,left=.08,right=.96,hspace=.65,wspace=.38)
        for ax,level in zip(axes.flat,levels):im=draw_panel(ax,scene,A,level)
        for ax in list(axes.flat)[len(levels):]:ax.set_visible(False)
        finish(fig,axes,im,f"HYPSO offset/damping optimum | {study['inputs'][scene]['label']} | A={A} DN\n"+settings,
               f"{scene}_A{A}_fig413")
    if 0 in levels:
        nrows=len(study["inputs"]);ncols=len(study["a_values"])
        fig,axes=plt.subplots(nrows,ncols,squeeze=False,figsize=(max(8.0,5.2*ncols),4.5*nrows+2.5))
        fig.subplots_adjust(top=.83 if nrows==1 else .91,bottom=.15,left=.08,right=.97,hspace=.7,wspace=.35)
        for i,scene in enumerate(study["inputs"]):
            for j,A in enumerate(study["a_values"]):im=draw_panel(axes[i,j],scene,A,0,True)
        finish(fig,axes,im,"HYPSO joint offset/damping grid | No added noise (existing sensor noise retained)\n"+settings,
               "all_images_no_added_noise")
