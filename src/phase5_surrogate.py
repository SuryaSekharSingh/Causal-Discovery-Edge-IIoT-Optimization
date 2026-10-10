"""Training-only grouped surrogate fitting and empirical residual calibration."""
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import TimeSeriesSplit

TARGETS = ["throughput_packets", "pdr", "drop_ratio", "latency_proxy_ms", "energy_proxy",
           "post_queue_server_a", "post_queue_server_b", "post_queue_server_c"]
QUEUES = ["pre_queue_server_a", "pre_queue_server_b", "pre_queue_server_c"]
ALLOCATIONS = ["allocation_server_a", "allocation_server_b", "allocation_server_c"]


def features(frame, buffer):
    return np.column_stack([frame.demand_packets.to_numpy(), frame[QUEUES].to_numpy()/buffer,
                            frame[ALLOCATIONS].to_numpy()])


def deduplicate(frame):
    keys = ["split", "graph_file"]+QUEUES+["action"]
    groups = frame.groupby(keys, sort=False)[TARGETS]
    spread = groups.max()-groups.min()
    if not np.allclose(spread,0,atol=1e-6,rtol=0):
        raise ValueError("Repeated input states have inconsistent outcomes")
    return frame.drop_duplicates(keys).copy()


def fit_surrogate(train, scm, config):
    if not train.split.eq("train").all():
        raise ValueError("Surrogate fitting requires training rows only")
    frame = deduplicate(train).sort_values(["window_start","graph_file","action"])
    graphs = frame.graph_file.drop_duplicates().tolist()
    cut = int(np.floor(len(graphs)*0.8))
    if cut<8 or cut>=len(graphs):
        raise ValueError("Insufficient training graphs for grouped temporal calibration")
    fitting = frame[frame.graph_file.isin(graphs[:cut])]
    calibration = frame[frame.graph_file.isin(graphs[cut:])]
    scale = np.array([scm.capacity.sum(),1,1,scm.budget,scm.max_energy,*scm.buffer])
    x = features(fitting,scm.buffer)
    mean, std = x.mean(0), x.std(0)
    std[std<1e-12]=1
    # Each fold fits its own input normalization; later statistics never enter CV.
    scores=[]
    for depth in (12,None):
        for leaf in (1,3):
            fold_scores=[]
            for fit_ids, val_ids in TimeSeriesSplit(n_splits=3).split(graphs[:cut]):
                f=fitting[fitting.graph_file.isin([graphs[i] for i in fit_ids])]
                v=fitting[fitting.graph_file.isin([graphs[i] for i in val_ids])]
                fx=features(f,scm.buffer); vx=features(v,scm.buffer)
                m=fx.mean(0); s=fx.std(0); s[s<1e-12]=1
                model=ExtraTreesRegressor(n_estimators=config["surrogate"]["n_estimators"],max_depth=depth,
                    min_samples_leaf=leaf,random_state=config["seed"],n_jobs=1)
                model.fit((fx-m)/s,f[TARGETS].to_numpy()/scale)
                fold_scores.append(float(np.mean((model.predict((vx-m)/s)-v[TARGETS].to_numpy()/scale)**2)))
            scores.append({"max_depth":depth,"min_samples_leaf":leaf,"fold_mse":fold_scores,"mean_mse":float(np.mean(fold_scores))})
    selected=min(scores,key=lambda s:s["mean_mse"])
    model=ExtraTreesRegressor(n_estimators=config["surrogate"]["n_estimators"],max_depth=selected["max_depth"],
        min_samples_leaf=selected["min_samples_leaf"],random_state=config["seed"],n_jobs=1)
    model.fit((x-mean)/std,fitting[TARGETS].to_numpy()/scale)
    errors=np.abs(model.predict((features(calibration,scm.buffer)-mean)/std)-calibration[TARGETS].to_numpy()/scale)
    # Group by physical state rather than rollout ID; all six actions share one band.
    c=calibration.copy(); c["_residual"]=errors.max(axis=1)
    residuals=c.groupby(["graph_file"]+QUEUES)["_residual"].max().to_numpy()
    rank=int(np.ceil((len(residuals)+1)*config["surrogate"]["coverage"]))
    if rank>len(residuals):
        raise ValueError("Insufficient calibration states for finite requested residual band")
    radius=float(np.sort(residuals)[rank-1])
    metadata={"fitted_on_split":"train","fit_graphs":graphs[:cut],"calibration_graphs":graphs[cut:],
              "fit_rows":len(fitting),"calibration_rows":len(calibration),"calibration_states":len(residuals),
              "cross_validation":scores,"selected":selected,"target_order":TARGETS,
              "input_order":["demand_packets"]+[q+"_ratio" for q in QUEUES]+ALLOCATIONS,
              "mean":mean.tolist(),"std":std.tolist(),"target_scale":scale.tolist(),
              "support_min":x[:,:4].min(0).tolist(),"support_max":x[:,:4].max(0).tolist(),
              "residual_radius":radius,"nominal_coverage":config["surrogate"]["coverage"],
              "interval_status":"empirical; temporal dependence prevents universal coverage guarantees",
              "controls_actions":False}
    return {"model":model,"metadata":metadata}, metadata


def predict_surrogate(bundle,frame,scm):
    meta=bundle["metadata"]; x=features(frame,scm.buffer)
    scale=np.asarray(meta["target_scale"])
    raw=bundle["model"].predict((x-meta["mean"])/meta["std"])*scale
    upper=np.array([scm.capacity.sum(),1,1,np.inf,scm.max_energy,*scm.buffer])
    bounded=np.minimum(np.maximum(raw,0),upper)
    supported=((x[:,:4]>=np.array(meta["support_min"])-1e-8)&(x[:,:4]<=np.array(meta["support_max"])+1e-8)).all(1)
    return raw,bounded,supported
