"""Standalone Phase 5 fitting, counterfactual replay, and artifact auditing."""
from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path
import shutil
import tempfile
import time
import importlib.metadata

import joblib
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.counterfactual_controller import (ActionAwareSCM, CounterfactualController, ControllerError,
    DecisionState, METRICS, load_phase5_scm)
from src.rl_environment import EdgeIIoTRoutingEnv, calibrate_environment, file_sha256, load_phase4_split
from src.rl_agent import load_phase4_agent
from src.phase5_surrogate import fit_surrogate, predict_surrogate, TARGETS, QUEUES, ALLOCATIONS

LOG=logging.getLogger(__name__)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path,payload):
    path=Path(path); temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(payload,indent=2,allow_nan=False),encoding="utf-8")
    temporary.replace(path)


def inputs(root,config):
    reports=root/config["phase4_reports_dir"]; artifacts=root/config["phase4_artifacts_dir"]
    manifest=read_json(reports/"run_manifest.json"); saved=manifest["config"]
    current=read_json(root/"config/phase4_config.json")
    if current!=saved:
        raise ValueError("Current Phase 4 configuration differs from authoritative saved configuration")
    paths={"causal_dataset_sha256":root/saved["causal_dataset_csv"],
           "phase3_encoder_sha256":root/saved["phase3_artifacts_dir"]/"encoder.pt"}
    paths.update({f"{s}_embeddings_sha256":root/saved["phase3_artifacts_dir"]/"embeddings"/f"{s}.pt" for s in ("train","validation","test")})
    paths.update({"best_model_sha256":artifacts/"best_model.zip","transition_dataset_sha256":root/saved["transition_dataset_csv"]})
    expected={**manifest["inputs"],**manifest["outputs"]}
    for key,path in paths.items():
        if file_sha256(path)!=expected[key]:
            raise ValueError(f"Phase 4 provenance mismatch: {key}")
    data={s:load_phase4_split(root,saved,s) for s in ("train","validation","test")}
    calibration=calibrate_environment(data["train"],saved)
    if read_json(artifacts/"environment_calibration.json")!=calibration.to_dict():
        raise ValueError("Saved calibration does not reproduce from training data")
    metadata=read_json(artifacts/"model_metadata.json")
    if metadata["observation"]["shape"]!=[67] or metadata["action_space"]["actions"]!=saved["environment"]["actions"] or not metadata["gnn_frozen"] or metadata["phase3_encoder_sha256"]!=data["train"].encoder_sha256:
        raise ValueError("Incompatible Phase 4 metadata")
    scm=ActionAwareSCM(calibration.to_dict(),saved["environment"])
    controller=CounterfactualController(scm,config["safety"])
    causal_features=read_json(root/"artifacts/phase2/refined_dag.json")["metadata"]["features"]
    if len(causal_features)!=9:
        raise ValueError("Expected nine Phase 2 features")
    if read_json(root/"reports/phase2/causal_variable_decisions.json")["used_features"]!=causal_features:
        raise ValueError("Phase 2 feature order disagrees with variable decisions")
    frame=pd.read_csv(paths["transition_dataset_sha256"],dtype={"would_truncate":str})
    required=set(["state_id","split","rollout_episode","step","graph_file","window_start","next_graph_file",
                  "action","action_name","would_truncate","demand_packets",*QUEUES,*ALLOCATIONS,*TARGETS,
                  "dropped_packets","task_completion","reward",*causal_features])
    if not required.issubset(frame):
        raise ValueError(f"Missing outcome columns: {required-set(frame)}")
    if frame[list(required)].isna().any().any() or not set(frame.split).issubset(data) or set(frame.split)!=set(data):
        raise ValueError("Missing values or invalid/missing outcome splits")
    boolean=frame.would_truncate.str.lower()
    if not boolean.isin(["true","false"]).all():
        raise ValueError("Invalid would_truncate boolean")
    frame["would_truncate"]=boolean.eq("true")
    numeric=frame.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Non-finite outcomes")
    if not frame.action.isin(range(6)).all():
        raise ValueError("Invalid action IDs")
    for name in ("step", "rollout_episode"):
        if not np.equal(frame[name], np.floor(frame[name])).all() or (frame[name]<0).any():
            raise ValueError(f"Invalid integer field: {name}")
    if (frame.step>=min(saved["environment"]["episode_horizon"],saved["transition_generation"]["episode_horizon"])).any():
        raise ValueError("Outcome step exceeds configured horizon")
    grouped=frame.groupby("state_id",sort=False)
    if not grouped.size().eq(6).all() or not grouped.action.nunique().eq(6).all():
        raise ValueError("Missing/duplicate state actions")
    invariants=["split","rollout_episode","step","graph_file","window_start","next_graph_file","would_truncate","demand_packets",*QUEUES,*causal_features]
    if not grouped[invariants].nunique(dropna=False).eq(1).all().all():
        raise ValueError("State rows disagree on pre-action context")
    partitions={s:frame[frame.split.eq(s)].copy() for s in data}
    max_error=0.0
    for split,part in partitions.items():
        source=data[split].frame.set_index("graph_file")
        times=pd.to_datetime(part.window_start,errors="raise")
        joined=source.loc[part.graph_file]
        if not np.array_equal(times.to_numpy(),joined.window_start.to_numpy()) or not np.allclose(part[causal_features],joined[causal_features],atol=1e-6,rtol=1e-8):
            raise ValueError("Outcome/causal identity mismatch")
        next_times=source.loc[part.next_graph_file].window_start.to_numpy()
        if not np.all(next_times-times.to_numpy()==np.timedelta64(60,"s")):
            raise ValueError("Outcome crosses timestamp gap")
        env=EdgeIIoTRoutingEnv(data[split],calibration,saved)
        indices={name:i for i,name in enumerate(data[split].frame.graph_file)}
        expected_truncation=np.array([int(step)+1>=min(saved["environment"]["episode_horizon"],saved["transition_generation"]["episode_horizon"]) or not env.valid_transition[indices[next_graph]]
            for step,next_graph in zip(part.step,part.next_graph_file)])
        if not np.array_equal(expected_truncation,part.would_truncate.to_numpy()):
            raise ValueError("Outcome truncation flag mismatch")
        if not np.allclose(part.demand_packets,part.total_packets,atol=1e-6,rtol=1e-8):
            raise ValueError("Demand differs from total_packets")
        for row in part.itertuples(index=False):
            state=DecisionState(row.demand_packets,tuple(getattr(row,k) for k in QUEUES),split,row.graph_file,row.window_start,scm.calibration_id)
            outcome=scm.intervene(state,int(row.action))
            if row.action_name!=outcome.action_name or not np.allclose([getattr(row,k) for k in ALLOCATIONS],outcome.allocation_fractions,atol=1e-10,rtol=0):
                raise ValueError("Action profile mismatch")
            for name in METRICS:
                error=abs(getattr(row,name)-getattr(outcome,name)); max_error=max(max_error,error)
                if not np.isclose(getattr(row,name),getattr(outcome,name),atol=1e-6,rtol=1e-8):
                    raise ValueError(f"Exact SCM mismatch: {split}/{row.state_id}/{name}")
            if not np.allclose([getattr(row,k) for k in TARGETS[-3:]],outcome.post_queue,atol=1e-6,rtol=1e-8):
                raise ValueError("Post-action queues disagree")
    starts=read_json(artifacts/"evaluation_start_states.json")
    for split in ("validation","test"):
        env=EdgeIIoTRoutingEnv(data[split],calibration,saved)
        if not starts[split] or any(i not in env.valid_starts for i in starts[split]):
            raise ValueError("Invalid saved evaluation starts")
    agent=load_phase4_agent(paths["best_model_sha256"],device="cpu")
    if agent.observation_space.shape!=(67,) or agent.action_space.n!=6:
        raise ValueError("DQN dimensions incompatible")
    provenance_paths=list(paths.values())+[reports/"run_manifest.json",artifacts/"environment_calibration.json",
        artifacts/"model_metadata.json",artifacts/"evaluation_start_states.json",root/"artifacts/phase2/refined_dag.json",
        root/"reports/phase2/causal_variable_decisions.json",root/"artifacts/phase2/causal_feature_mapping.json",
        root/saved["phase3_artifacts_dir"]/"model_metadata.json",root/"config/phase4_config.json",
        root/"src/counterfactual_controller.py",root/"src/phase5_surrogate.py",root/"src/phase5_pipeline.py",
        root/"src/rl_environment.py",root/"src/rl_agent.py"]
    hashes={str(p.relative_to(root)):file_sha256(p) for p in provenance_paths}
    audit={"rows":len(frame),"rows_by_split":{s:len(p) for s,p in partitions.items()},
           "decision_states":int(frame.state_id.nunique()),"exact_scm_max_abs_error":max_error,
           "split_counts":{s:d.size for s,d in data.items()},"all_checks_passed":True,
           "singleton_snapshots":{s:int((~EdgeIIoTRoutingEnv(d,calibration,saved).valid_transition &
               ~np.r_[False,EdgeIIoTRoutingEnv(d,calibration,saved).valid_transition[:-1]]).sum()) for s,d in data.items()}}
    return saved,data,calibration,scm,controller,partitions,starts,agent,hashes,audit


def surrogate_reports(bundle,parts,scm,controller,reports):
    summary={}; records=[]
    for split,frame in parts.items():
        raw,bounded,supported=predict_surrogate(bundle,frame,scm)
        truth=frame[TARGETS].to_numpy(); scale=np.array(bundle["metadata"]["target_scale"])
        radius=bundle["metadata"]["residual_radius"]
        covered=(np.abs(raw-truth)/scale<=radius+1e-12).all(1)
        true_safe=(truth[:,3]<=controller.limits["latency_proxy_ms"]+controller.tolerance)&(truth[:,4]<=controller.limits["energy_proxy"]+controller.tolerance)&(truth[:,2]<=controller.limits["drop_ratio"]+controller.tolerance)&(truth[:,1]>=controller.limits["pdr"]-controller.tolerance)
        learned_safe=(bounded[:,3]<=controller.limits["latency_proxy_ms"]+controller.tolerance)&(bounded[:,4]<=controller.limits["energy_proxy"]+controller.tolerance)&(bounded[:,2]<=controller.limits["drop_ratio"]+controller.tolerance)&(bounded[:,1]>=controller.limits["pdr"]-controller.tolerance)
        state_coverage=pd.Series(covered,index=frame.index).groupby(frame.state_id).all().mean()
        metrics={}
        for mode,pred in (("raw",raw),("bounded",bounded)):
            metrics[mode]={name:{"mae":float(mean_absolute_error(truth[:,i],pred[:,i])),
                                "rmse":float(np.sqrt(mean_squared_error(truth[:,i],pred[:,i]))),
                                "r2":float(r2_score(truth[:,i],pred[:,i])),
                                "normalized_mse":float(np.mean(((truth[:,i]-pred[:,i])/scale[i])**2))}
                           for i,name in enumerate(TARGETS)}
        # Current service is demand*PDR; old service is determined by pre-action queues.
        old=np.minimum(frame[QUEUES].to_numpy(),scm.capacity).sum(1)
        conservation=np.abs(raw[:,0]-old-frame.demand_packets.to_numpy()*raw[:,1])
        packet_conservation={mode:float(np.mean(np.abs(frame[QUEUES].sum(axis=1).to_numpy()+frame.demand_packets.to_numpy()-pred[:,0]-pred[:,5:].sum(1)-frame.demand_packets.to_numpy()*pred[:,2])))
            for mode,pred in (("raw",raw),("bounded",bounded))}
        summary[split]={"metrics":metrics,"empirical_state_coverage":float(state_coverage),
                        "unsupported_fraction":float((~supported).mean()),
                        "false_safe_fraction":float((learned_safe&~true_safe).mean()),
                        "false_unsafe_fraction":float((~learned_safe&true_safe).mean()),
                        "raw_throughput_conservation_mae":float(conservation.mean()),
                        "packet_conservation_mae":packet_conservation,
                        "qualification":"diagnostic_only","controls_actions":False,
                        "quality_flags":(["uninformative_residual_band"] if radius>1 else [])+
                            (["unsupported_states_present"] if not supported.all() else [])+
                            (["coverage_below_nominal"] if state_coverage<bundle["metadata"]["nominal_coverage"] else []),
                        "normalized_residual_radius":radius,
                        "interval_full_width_by_target":dict(zip(TARGETS,(2*radius*scale).tolist()))}
        r=frame[["state_id","split","action"]].reset_index(drop=True).copy()
        r["supported"]=supported; r["all_targets_covered"]=covered
        r["exact_safe"]=true_safe; r["surrogate_safe"]=learned_safe
        for i,name in enumerate(TARGETS):
            r[name+"_actual"]=truth[:,i]; r[name+"_raw"]=raw[:,i]; r[name+"_bounded"]=bounded[:,i]
            r[name+"_lower"]=raw[:,i]-radius*scale[i]; r[name+"_upper"]=raw[:,i]+radius*scale[i]
        records.append(r)
    pd.concat(records).to_csv(reports/"surrogate_predictions.csv",index=False)
    write_json(reports/"surrogate_metrics.json",summary)
    return summary


def evaluate(data,calibration,saved,starts,agent,controller,reports):
    episodes=[]; actions=[]; summary={}
    with (reports/"explanations.jsonl").open("w",encoding="utf-8") as explanations:
        for split in ("validation","test"):
            summary[split]={}
            for policy in ("dqn","screened_dqn","least_queue"):
                step_records=[]
                for episode,start in enumerate(starts[split]):
                    env=EdgeIIoTRoutingEnv(data[split],calibration,saved)
                    obs,_=env.reset(seed=42+episode,options={"start_index":start})
                    cumulative=0; totals={k:0.0 for k in METRICS if k!="reward"}; step=0; done=False
                    while not done:
                        started=time.perf_counter()
                        if policy=="screened_dqn":
                            result,decision,state=controller.step(env,agent)
                            detail=decision.to_dict(); detail.update({"state":as_state(state),"episode":episode,"step":step,
                                "realized":{k:float(result[4][k]) for k in METRICS},"realized_post_queue":result[4]["post_queue"].tolist()})
                            explanations.write(json.dumps(detail,allow_nan=False)+"\n")
                            for a,o in enumerate(decision.outcomes):
                                actions.append({"split":split,"episode":episode,"step":step,"graph_file":state.graph_file,
                                    "action":a,"action_name":o.action_name,"q_value":decision.q_values[a],
                                    "rank":decision.ranking.index(a),"proposed":a==decision.proposed_action,
                                    "selected":a==decision.executed_action,"safe":not decision.reasons[a],
                                    "reasons":";".join(decision.reasons[a]),"violation_score":decision.violation_scores[a],
                                    "status":decision.status,"safety_feasible":decision.safety_feasible,
                                    **{k:getattr(o,k) for k in METRICS}})
                            chosen=decision.outcomes[decision.executed_action]; proposed=decision.outcomes[decision.proposed_action]
                            diagnostics={"intervened":decision.proposed_action!=decision.executed_action,
                                "feasible":decision.safety_feasible,"violated":bool(decision.reasons[decision.executed_action]),
                                "no_safe_action":not decision.safety_feasible,"query_seconds":decision.query_seconds,
                                "q_loss":decision.q_values[decision.proposed_action]-decision.q_values[decision.executed_action],
                                "rank":decision.ranking.index(decision.executed_action),
                                "veto_fraction":sum(bool(r) for r in decision.reasons)/6,
                                **{k+"_delta":getattr(chosen,k)-getattr(proposed,k) for k in METRICS}}
                        else:
                            if policy=="dqn":
                                a,_=agent.predict(obs,deterministic=True); a=int(np.asarray(a).item())
                            else:
                                a=int(np.argmin(obs[-3:]))
                            state=DecisionState(float(env.current_row.total_packets),tuple(env.queues),split,
                                str(env.current_row.graph_file),env.current_row.window_start.isoformat(),controller.scm.calibration_id)
                            predictions=controller.scm.evaluate_actions(state)
                            assessments=[controller.assess(o)[0] for o in predictions]
                            diagnostics={"feasible":any(not r for r in assessments),"violated":bool(assessments[a])}
                            result=env.step(a)
                        obs,reward,terminated,truncated,info=result; done=terminated or truncated
                        diagnostics.update({"action":int(info["action"]),"controller_seconds":time.perf_counter()-started,
                            **{"actual_"+k:float(info[k]) for k in METRICS}})
                        step_records.append(diagnostics)
                        cumulative+=reward
                        for k in totals: totals[k]+=info[k]
                        step+=1
                    episodes.append({"split":split,"policy":policy,"episode":episode,"start_index":start,
                        "steps":step,"cumulative_reward":cumulative,"mean_step_reward":cumulative/step,
                        **{k:v/step for k,v in totals.items()}})
                subset=[r for r in episodes if r["split"]==split and r["policy"]==policy]
                agg={k:float(np.mean([r[k] for r in subset])) for k in ["cumulative_reward","mean_step_reward",*totals]}
                agg.update({"steps":len(step_records),"feasible_fraction":float(np.mean([r["feasible"] for r in step_records])),
                    "violations_on_feasible_states":sum(r["feasible"] and r["violated"] for r in step_records),
                    "unavoidable_violations":sum(not r["feasible"] and r["violated"] for r in step_records),
                    "action_counts":np.bincount([r["action"] for r in step_records],minlength=6).tolist()})
                agg["metric_aggregation"]="mean of episode means; cumulative reward is mean episode total"
                agg["step_weighted_metrics"]={k:float(np.mean([r["actual_"+k] for r in step_records])) for k in METRICS}
                if policy=="screened_dqn":
                    if agg["violations_on_feasible_states"]: raise RuntimeError("Screening violated a feasible state")
                    for k in ("intervened","no_safe_action","query_seconds","controller_seconds","q_loss","rank","veto_fraction"):
                        agg[k+"_mean"]=float(np.mean([r[k] for r in step_records]))
                    agg["query_seconds_max"]=max(r["query_seconds"] for r in step_records)
                    agg["same_state_deltas"]={k:float(np.mean([r[k+"_delta"] for r in step_records])) for k in METRICS}
                summary[split][policy]=agg
    pd.DataFrame(episodes).to_csv(reports/"evaluation_episodes.csv",index=False)
    pd.DataFrame(actions).to_csv(reports/"decision_actions.csv",index=False)
    write_json(reports/"evaluation_metrics.json",summary)
    distribution=[{"split":s,"policy":p,"action":a,"count":n} for s,policies in summary.items() for p,m in policies.items() for a,n in enumerate(m["action_counts"])]
    pd.DataFrame(distribution).to_csv(reports/"action_distribution.csv",index=False)
    write_json(reports/"violation_summary.json",{s:{p:{k:v for k,v in m.items() if "violation" in k or "feasible" in k or "no_safe" in k} for p,m in policies.items()} for s,policies in summary.items()})
    return pd.DataFrame(episodes),summary


def as_state(state):
    from dataclasses import asdict
    return asdict(state)


def bootstrap(episodes,config):
    rng=np.random.default_rng(config["seed"]); results={}
    for split in ("validation","test"):
        part=episodes[episodes.split.eq(split)]; base=part[part.policy.eq("dqn")].set_index("episode")
        screened=part[part.policy.eq("screened_dqn")].set_index("episode").loc[base.index]
        results[split]={}
        for metric in ("cumulative_reward","latency_proxy_ms","energy_proxy","drop_ratio","pdr"):
            differences=(screened[metric]-base[metric]).to_numpy()
            samples=rng.choice(differences,size=(config["bootstrap_resamples"],len(differences)),replace=True).mean(1)
            results[split][metric]={"mean_difference":float(differences.mean()),"interval_95":np.quantile(samples,[.025,.975]).tolist()}
    return {"status":"descriptive; evaluation episodes may overlap","paired_differences":results}


def plots(reports,summary):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for name,metrics in (("policy_comparison",["cumulative_reward","latency_proxy_ms","energy_proxy"]),
                             ("safety_comparison",["feasible_fraction","violations_on_feasible_states","unavoidable_violations"])):
            fig,axes=plt.subplots(1,3,figsize=(14,4))
            labels=list(summary["test"])
            for ax,metric in zip(axes,metrics):
                ax.bar(labels,[summary["test"][p][metric] for p in labels]); ax.set_title(metric); ax.tick_params(axis="x",rotation=20)
            fig.tight_layout(); fig.savefig(reports/(name+".png"),dpi=160); plt.close(fig)
        data=pd.read_csv(reports/"surrogate_predictions.csv"); test=data[data.split.eq("test")]
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for ax,k in zip(axes,["latency_proxy_ms","energy_proxy"]):
            ax.scatter(test[k+"_actual"],test[k+"_raw"],s=4,alpha=.25); ax.set_xlabel("Actual "+k); ax.set_ylabel("Surrogate prediction")
            if k=="latency_proxy_ms": ax.ticklabel_format(axis="both",style="sci",scilimits=(0,0))
        fig.tight_layout(); fig.savefig(reports/"prediction_quality.png",dpi=160); plt.close(fig)
        return {"status":"complete","files":["policy_comparison.png","safety_comparison.png","prediction_quality.png"]}
    except Exception as error:
        LOG.warning("Optional plotting failed: %s",error)
        return {"status":"warning","error":str(error),"files":[]}


def audit_outputs(root,artifacts,reports):
    manifest=read_json(reports/"run_manifest.json")
    for path,digest in manifest["inputs"].items():
        if file_sha256(root/path)!=digest: raise ValueError(f"Changed input: {path}")
    for key,folder in (("artifacts",artifacts),("reports",reports)):
        for name,digest in manifest["outputs"][key].items():
            if file_sha256(folder/name)!=digest: raise ValueError(f"Changed output: {name}")
    required_artifacts=["exact_scm.json","action_dag.json","surrogate.joblib","surrogate_metadata.json","residual_calibration.json","safety_config.json","controller_metadata.json","evaluation_start_states.json","completion_status.json"]
    required_reports=["input_audit.json","surrogate_metrics.json","surrogate_predictions.csv","evaluation_episodes.csv","evaluation_metrics.json","decision_actions.csv","explanations.jsonl","action_distribution.csv","violation_summary.json","bootstrap_intervals.json","plot_status.json"]
    for folder,names in ((artifacts,required_artifacts),(reports,required_reports)):
        for name in names:
            path=folder/name
            if not path.is_file() or not path.stat().st_size: raise ValueError(f"Missing artifact: {path}")
            if path.suffix==".json": read_json(path)
            if path.suffix==".csv" and pd.read_csv(path).empty: raise ValueError(f"Empty CSV: {path}")
    scm=load_phase5_scm(artifacts); controller=CounterfactualController(scm,read_json(artifacts/"safety_config.json"))
    bundle=joblib.load(artifacts/"surrogate.joblib")
    if bundle["metadata"]!=read_json(artifacts/"surrogate_metadata.json"): raise ValueError("Surrogate metadata mismatch")
    model=bundle["model"]
    if not np.isfinite(model.predict(np.zeros((1,7)))).all(): raise ValueError("Surrogate reload failed")
    original=pd.read_csv(root/manifest["phase4_config"]["transition_dataset_csv"])
    sample=pd.read_csv(reports/"surrogate_predictions.csv",nrows=128)
    joined=original.set_index(["state_id","action"]).loc[list(zip(sample.state_id,sample.action))]
    raw,_,_=predict_surrogate(bundle,joined,scm)
    if not np.allclose(raw,sample[[k+"_raw" for k in TARGETS]],atol=1e-6,rtol=1e-8):
        raise ValueError("Reloaded surrogate predictions differ from export")
    records=0
    with (reports/"explanations.jsonl").open() as handle:
        for line in handle:
            record=json.loads(line); state=DecisionState(**{**record["state"],"queues":tuple(record["state"]["queues"])})
            decision=controller.select_action(np.zeros(67),state,record["q_values"])
            if decision.executed_action!=record["executed_action"] or decision.status!=record["status"] or len(record["outcomes"])!=6:
                raise ValueError("Explanation selection audit failed")
            for actual,stored in zip(decision.outcomes,record["outcomes"]):
                if any(not np.isclose(getattr(actual,k),stored[k],atol=1e-6,rtol=1e-8) for k in METRICS): raise ValueError("Explanation prediction mismatch")
            for k in METRICS:
                if not np.isclose(getattr(decision.outcomes[decision.executed_action],k),record["realized"][k],atol=1e-6,rtol=1e-8): raise ValueError("Realized outcome audit failed")
            if not np.allclose(decision.outcomes[decision.executed_action].post_queue,record["realized_post_queue"],atol=1e-6,rtol=1e-8): raise ValueError("Realized queue audit failed")
            records+=1
    actions=pd.read_csv(reports/"decision_actions.csv")
    groups=actions.groupby(["split","episode","step"])
    if len(groups)!=records or not groups.size().eq(6).all() or not groups.action.nunique().eq(6).all() or not groups.selected.sum().eq(1).all():
        raise ValueError("Incomplete decision/action coverage")
    if not actions.loc[actions.selected & actions.safety_feasible,"safe"].all(): raise ValueError("Unsafe feasible decision")
    plot_status=read_json(reports/"plot_status.json")
    if plot_status["status"]=="complete":
        from PIL import Image
        for name in plot_status["files"]:
            with Image.open(reports/name) as image: image.verify()
    status=read_json(artifacts/"completion_status.json")
    if status["status"]!="complete": raise ValueError("Run incomplete")
    result={"status":"passed","screened_decisions":records,"all_mandatory_checks_passed":True}
    LOG.info("ALL_PHASE5_CHECKS_PASSED")
    return result


def execute(root,config,artifacts,reports,smoke=False):
    saved,data,calibration,scm,controller,parts,starts,agent,hashes,input_audit=inputs(root,config)
    write_json(reports/"input_audit.json",input_audit)
    write_json(artifacts/"exact_scm.json",scm.specification())
    edges=[("action","allocation"),("allocation","arrivals"),("demand","arrivals"),
           ("queues","old_service"),("capacity","old_service"),("old_service","remaining_capacity"),
           ("capacity","remaining_capacity"),("queues","residual_queue"),("old_service","residual_queue"),
           ("arrivals","current_service"),("remaining_capacity","current_service"),
           ("arrivals","unserved"),("current_service","unserved"),("unserved","post_queue"),
           ("buffer","post_queue"),("residual_queue","post_queue"),("unserved","drops"),
           ("buffer","drops"),("residual_queue","drops"),("current_service","throughput"),
           ("old_service","throughput"),("current_service","pdr"),("demand","pdr"),
           ("queues","latency_proxy_ms"),("capacity","latency_proxy_ms"),("base_latency","latency_proxy_ms"),
           ("allocation","latency_proxy_ms"),("pdr","latency_proxy_ms"),
           ("queues","energy_proxy"),("arrivals","energy_proxy"),("old_service","energy_proxy"),
           ("current_service","energy_proxy"),("capacity","energy_proxy"),("energy_coefficients","energy_proxy"),
           ("pdr","completion"),("latency_proxy_ms","completion"),
           ("drops","drop_ratio"),("demand","drop_ratio")]
    edges += [(k,"reward") for k in ("throughput","pdr","completion","latency_proxy_ms","energy_proxy","drop_ratio")]
    edges += [("window_seconds","latency_proxy_ms"),("latency_budget_ms","latency_proxy_ms"),
              ("latency_budget_ms","completion"),("pdr_completion_threshold","completion"),
              ("reward_weights","reward"),("capacity","reward"),("energy_coefficients","reward"),
              ("latency_budget_ms","reward")]
    graph=nx.DiGraph(edges)
    if not nx.is_directed_acyclic_graph(graph): raise RuntimeError("Action DAG cyclic")
    write_json(artifacts/"action_dag.json",{"nodes":list(graph.nodes),"edges":edges,"scope":"deterministic Phase 4 simulator"})
    write_json(artifacts/"safety_config.json",config["safety"])
    if smoke:
        graphs=parts["train"].sort_values("window_start").graph_file.drop_duplicates().tolist()[:120]
        train=parts["train"][parts["train"].graph_file.isin(graphs)]
        config=copy.deepcopy(config); config["surrogate"]["n_estimators"]=10
        starts={s:v[:1] for s,v in starts.items()}
        saved=copy.deepcopy(saved); saved["environment"]["episode_horizon"]=min(4,saved["environment"]["episode_horizon"])
        scm=ActionAwareSCM(calibration.to_dict(),saved["environment"]); controller=CounterfactualController(scm,config["safety"])
        write_json(artifacts/"exact_scm.json",scm.specification())
    else: train=parts["train"]
    LOG.info("Fitting grouped training-only outcome surrogate.")
    bundle,metadata=fit_surrogate(train,scm,config)
    joblib.dump(bundle,artifacts/"surrogate.joblib")
    write_json(artifacts/"surrogate_metadata.json",metadata)
    write_json(artifacts/"residual_calibration.json",{k:metadata[k] for k in ("residual_radius","nominal_coverage","support_min","support_max","calibration_states","interval_status")})
    write_json(artifacts/"evaluation_start_states.json",starts)
    write_json(artifacts/"controller_metadata.json",{"schema_version":1,"backend":"exact_simulator_scm",
        "surrogate_controls_actions":False,"one_step_only":True,"state_information":["actual_demand","actual_queues","frozen_embedding"],
        "resolved_limits":controller.limits,"seed":config["seed"],"test_usage":"evaluation after surrogate and safety configuration frozen",
        "metric_status":{"latency_proxy_ms":"synthetic","energy_proxy":"synthetic"}})
    surrogate_reports(bundle,parts,scm,controller,reports)
    LOG.info("Evaluating frozen policies and recording six-action explanations.")
    episodes,summary=evaluate(data,calibration,saved,starts,agent,controller,reports)
    write_json(reports/"bootstrap_intervals.json",bootstrap(episodes,config))
    plot_status=plots(reports,summary) if config["generate_plots"] else {"status":"disabled","files":[]}
    write_json(reports/"plot_status.json",plot_status)
    write_json(artifacts/"completion_status.json",{"status":"complete","smoke_test":smoke,"surrogate_qualification":"diagnostic_only","plots":plot_status["status"]})
    for name,digest in hashes.items():
        if file_sha256(root/name)!=digest: raise ValueError("Upstream input changed during Phase 5")
    output_hashes={key:{p.name:file_sha256(p) for p in folder.iterdir() if p.is_file() and p.name not in {"run_manifest.json","artifact_audit.json"}}
                   for key,folder in (("artifacts",artifacts),("reports",reports))}
    versions={p:importlib.metadata.version(p) for p in ("numpy","pandas","torch","scikit-learn","stable-baselines3","joblib")}
    write_json(reports/"run_manifest.json",{"config":config,"phase4_config":saved,"inputs":hashes,"outputs":output_hashes,"package_versions":versions})
    write_json(reports/"artifact_audit.json",audit_outputs(root,artifacts,reports))
    manifest=read_json(reports/"run_manifest.json")
    manifest["outputs"]["reports"]["artifact_audit.json"]=file_sha256(reports/"artifact_audit.json")
    write_json(reports/"run_manifest.json",manifest)


def run(config_path,validate_only=False,smoke_test=False,audit_only=False):
    root=Path.cwd().resolve(); config=read_json(config_path)
    artifacts=(root/config["artifacts_dir"]).resolve(); reports=(root/config["reports_dir"]).resolve()
    for folder,parent in ((artifacts,root/"artifacts"),(reports,root/"reports")):
        if folder.parent!=parent.resolve() or folder.name!="phase5": raise ValueError("Outputs must be confined to Phase 5 directories")
    if audit_only:
        audit_outputs(root,artifacts,reports); return
    if validate_only:
        *_,audit=inputs(root,config); LOG.info("PHASE5_INPUT_CHECKS_PASSED: %s",audit); return
    if smoke_test:
        with tempfile.TemporaryDirectory(prefix="phase5_smoke_") as temporary:
            a=Path(temporary)/"artifacts"; r=Path(temporary)/"reports"; a.mkdir(); r.mkdir()
            execute(root,config,a,r,smoke=True)
        LOG.info("PHASE5_SMOKE_TEST_PASSED"); return
    for folder in (artifacts,reports):
        if folder.exists() and any(folder.iterdir()) and not config["overwrite_outputs"]:
            raise FileExistsError(f"Refusing non-empty output: {folder}")
    with tempfile.TemporaryDirectory(prefix="phase5_staging_",dir=root) as temporary:
        a=Path(temporary)/"artifacts"; r=Path(temporary)/"reports"; a.mkdir(); r.mkdir()
        try:
            execute(root,config,a,r)
        except Exception as error:
            write_json(r/"failure.json",error.to_dict() if isinstance(error,ControllerError) else {"error":str(error)})
            failure=Path(tempfile.mkdtemp(prefix="phase5_failed_",dir=root))
            shutil.copytree(a,failure/"artifacts"); shutil.copytree(r,failure/"reports")
            LOG.error("Failure diagnostics preserved: %s",failure)
            raise
        for staged,destination in ((a,artifacts),(r,reports)):
            if destination.exists(): shutil.rmtree(destination)
            shutil.move(str(staged),str(destination))
    LOG.info("Phase 5 complete: %s",artifacts)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="config/phase5_config.json",type=Path)
    mode=parser.add_mutually_exclusive_group()
    for flag in ("validate-only","smoke-test","audit-only"): mode.add_argument("--"+flag,action="store_true")
    args=parser.parse_args(); logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
    run(args.config,args.validate_only,args.smoke_test,args.audit_only)


if __name__=="__main__": main()
