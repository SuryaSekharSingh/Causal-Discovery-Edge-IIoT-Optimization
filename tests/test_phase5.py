"""Independent SCM and decision-rule regression tests."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.counterfactual_controller import (ActionAwareSCM, CounterfactualController,
    ControllerError, DecisionState, METRICS)
from src.rl_environment import EdgeIIoTRoutingEnv, EnvironmentCalibration, Phase4SplitData
from src.phase5_surrogate import deduplicate, fit_surrogate
from src.phase5_pipeline import inputs
from src.rl_agent import load_phase4_agent


class Phase5Tests(unittest.TestCase):
    def setUp(self):
        manifest=json.loads(Path("reports/phase4/run_manifest.json").read_text())
        self.config=manifest["config"]
        calibration=json.loads(Path("artifacts/phase4/environment_calibration.json").read_text())
        self.scm=ActionAwareSCM(calibration,self.config["environment"])
        self.safety=json.loads(Path("config/phase5_config.json").read_text())["safety"]
        self.controller=CounterfactualController(self.scm,self.safety)
        self.state=DecisionState(317,(0,0,0),"train","graph.pt","2021-01-01T00:00:00",self.scm.calibration_id)

    def test_hand_calculation_and_zero_demand(self):
        result=self.scm.intervene(self.state,0)
        self.assertEqual(result.throughput_packets,317)
        self.assertEqual(result.pdr,1)
        self.assertEqual(result.latency_proxy_ms,5)
        self.assertAlmostEqual(result.energy_proxy,1+2.5*(317/self.scm.capacity[0])**2)
        zero=self.scm.intervene(replace(self.state,demand_packets=0),0)
        self.assertEqual(zero.pdr,1); self.assertEqual(zero.energy_proxy,0)
        self.assertEqual(zero.drop_ratio,0); self.assertEqual(zero.post_queue,(0,0,0))

    def test_all_profiles_conserve_packets(self):
        for demand in (0,300,1400,100000):
            for queue in ((0,0,0),tuple(self.scm.buffer)):
                state=replace(self.state,demand_packets=demand,queues=queue)
                for o in self.scm.evaluate_actions(state):
                    intermediate=dict(o.intermediates)
                    self.assertAlmostEqual(sum(queue)+demand,o.throughput_packets+sum(o.post_queue)+o.dropped_packets,places=7)
                    self.assertTrue(np.all(np.array(o.post_queue)<=self.scm.buffer))
                    self.assertTrue(np.all(np.array(o.post_queue)>=0))
                    self.assertTrue(np.isfinite([getattr(o,k) for k in METRICS]).all())
                    self.assertTrue(np.all(np.array(intermediate["served_current"])<=np.array(intermediate["arrivals"])))

    def test_ranking_and_correction(self):
        observation=np.zeros(67)
        q=np.array([6,5,4,3,2,1])
        d=self.controller.select_action(observation,self.state,q)
        self.assertEqual(d.executed_action,0); self.assertEqual(d.status,"unchanged")
        state=replace(self.state,demand_packets=700)
        q=np.array([5,6,4,3,2,1])
        d=self.controller.select_action(observation,state,q)
        self.assertEqual(d.proposed_action,1); self.assertEqual(d.executed_action,0)
        self.assertEqual(d.status,"corrected")
        tied=self.controller.select_action(observation,self.state,np.ones(6))
        self.assertEqual(tied.proposed_action,0)

    def test_no_safe_fallback_and_thresholds(self):
        d=self.controller.select_action(np.zeros(67),replace(self.state,demand_packets=100000),np.arange(6))
        self.assertFalse(d.safety_feasible); self.assertEqual(d.status,"no_safe_action")
        expected=min(range(6),key=lambda a:(d.violation_scores[a],-d.q_values[a],a))
        self.assertEqual(expected,d.executed_action)
        o=self.scm.intervene(self.state,0)
        boundary=replace(o,latency_proxy_ms=15000,energy_proxy=self.controller.limits["energy_proxy"],drop_ratio=.01,pdr=.99)
        self.assertEqual(self.controller.assess(boundary)[0],())
        self.assertIn("pdr",self.controller.assess(replace(boundary,pdr=.99-2e-8))[0])

    def test_reject_invalid_inputs(self):
        for state in (replace(self.state,demand_packets=-1),replace(self.state,queues=(-1,0,0)),
                      replace(self.state,calibration_id="stale"),replace(self.state,demand_packets=np.nan)):
            with self.assertRaises(ControllerError): self.scm.evaluate_actions(state)
        for action in (-1,6,1.5,True):
            with self.assertRaises(ControllerError): self.scm.intervene(self.state,action)
        with self.assertRaises(ControllerError): self.controller.select_action(np.zeros(67),self.state,[np.nan]*6)
        with self.assertRaises(ControllerError): self.controller.select_action(["invalid"]*67,self.state,np.ones(6))
        with self.assertRaises(ControllerError): self.scm.intervene(None,0)
        with self.assertRaises(ControllerError): self.scm.intervene(replace(self.state,demand_packets="invalid"),0)
        with self.assertRaises(ControllerError): self.controller.assess(replace(self.scm.intervene(self.state,0),energy_proxy=np.inf))

    def test_independent_environment_agreement_and_query_purity(self):
        c=self.scm.calibration
        calibration=EnvironmentCalibration(**{k:np.array(v) if isinstance(v,list) else v for k,v in c.items()
            if k not in {"server_names","metric_status"}})
        frame=pd.DataFrame({"window_start":pd.date_range("2021-01-01",periods=3,freq="60s"),
                            "graph_file":["a","b","c"],"total_packets":[700,100000,0]})
        data=Phase4SplitData("train",np.zeros((3,64),dtype=np.float32),frame,"hash")
        env=EdgeIIoTRoutingEnv(data,calibration,self.config)
        env.reset(seed=42,options={"start_index":0,"queues":[1,2,3]})
        rng=copy.deepcopy(env.np_random.bit_generator.state)
        state=replace(self.state,demand_packets=700,queues=(1,2,3))
        for o in self.scm.evaluate_actions(state):
            expected=env.simulate_action(o.action)
            for k in METRICS: self.assertAlmostEqual(getattr(o,k),getattr(expected,k),places=8)
            np.testing.assert_allclose(o.post_queue,expected.post_queue,atol=1e-8)
        self.assertEqual(env.current_index,0); self.assertEqual(env.episode_steps,0)
        np.testing.assert_array_equal(env.queues,[1,2,3])
        self.assertEqual(rng,env.np_random.bit_generator.state)

    def test_training_only_and_duplicate_conflict(self):
        frame=pd.read_csv("data/interim/phase4_action_outcomes.csv")
        with self.assertRaises(ValueError): fit_surrogate(frame[frame.split.eq("test")],self.scm,{})
        example=frame.iloc[:6].copy()
        duplicate=example.iloc[[0]].copy(); duplicate["pdr"]-=.1
        with self.assertRaises(ValueError): deduplicate(pd.concat([example,duplicate]))
        self.assertEqual(len(deduplicate(pd.concat([example,example]))),6)

    def test_input_rejection_cases(self):
        original_read=pd.read_csv
        frame=original_read("data/interim/phase4_action_outcomes.csv",dtype={"would_truncate":str})
        frame=pd.concat([frame[frame.split.eq(s)].iloc[:6] for s in ("train","validation","test")]).reset_index(drop=True)
        config=json.loads(Path("config/phase5_config.json").read_text())
        variants=[]
        duplicate=frame.copy(); duplicate.loc[1,"action"]=0; variants.append(duplicate)
        boolean=frame.copy(); boolean.loc[0,"would_truncate"]="invalid"; variants.append(boolean)
        mismatch=frame.copy(); mismatch.loc[:5,"window_start"]="2021-01-01T00:00:00"; variants.append(mismatch)
        queues=frame.copy(); queues.loc[0,"pre_queue_server_a"]=-1; variants.append(queues)
        nonfinite=frame.copy(); nonfinite.loc[0,"pdr"]=np.inf; variants.append(nonfinite)
        truncation=frame.copy(); truncation.loc[:5,"would_truncate"]="True"; variants.append(truncation)
        variants.append(frame.drop(columns=["action_name"]))
        for malformed in variants:
            def replacement(path,*args,**kwargs):
                return malformed.copy() if Path(path).name=="phase4_action_outcomes.csv" else original_read(path,*args,**kwargs)
            with patch("src.phase5_pipeline.pd.read_csv",side_effect=replacement):
                with self.assertRaises((ValueError,KeyError)):
                    inputs(Path.cwd(),config)
        with patch("src.phase5_pipeline.file_sha256",return_value="stale"):
            with self.assertRaisesRegex(ValueError,"provenance"):
                inputs(Path.cwd(),config)

    def test_runtime_single_execution_and_frozen_agent(self):
        from src.rl_environment import load_phase4_split, calibrate_environment
        data=load_phase4_split(Path.cwd(),self.config,"validation")
        train=load_phase4_split(Path.cwd(),self.config,"train")
        calibration=calibrate_environment(train,self.config)
        env=EdgeIIoTRoutingEnv(data,calibration,self.config)
        agent=load_phase4_agent("artifacts/phase4/best_model.zip",device="cpu")
        before={k:v.detach().clone() for k,v in agent.policy.state_dict().items()}
        env.reset(seed=42,options={"start_index":0})
        with patch.object(env,"step",wraps=env.step) as step:
            result,decision,state=self.controller.step(env,agent)
            self.assertEqual(step.call_count,1)
            self.assertEqual(env.episode_steps,1)
            np.testing.assert_allclose(result[4]["post_queue"],decision.outcomes[decision.executed_action].post_queue)
        for k,v in agent.policy.state_dict().items():
            np.testing.assert_array_equal(before[k].numpy(),v.detach().numpy())
        env.episode_steps=env.episode_horizon
        with patch.object(env,"step",wraps=env.step) as step:
            with self.assertRaises(ControllerError): self.controller.step(env,agent)
            self.assertEqual(step.call_count,0)

    def test_grouped_fit_and_calibration_disjoint(self):
        frame=pd.read_csv("data/interim/phase4_action_outcomes.csv")
        train=frame[frame.split.eq("train")]
        graphs=train.sort_values("window_start").graph_file.drop_duplicates().tolist()[:120]
        config=json.loads(Path("config/phase5_config.json").read_text()); config["surrogate"]["n_estimators"]=2
        bundle,metadata=fit_surrogate(train[train.graph_file.isin(graphs)],self.scm,config)
        self.assertFalse(set(metadata["fit_graphs"]) & set(metadata["calibration_graphs"]))
        self.assertEqual(metadata["fit_graphs"],graphs[:96])
        self.assertEqual(metadata["calibration_graphs"],graphs[96:])
        self.assertEqual(bundle["model"].n_features_in_,7)
        self.assertGreaterEqual(metadata["calibration_states"],19)


if __name__=="__main__": unittest.main()
