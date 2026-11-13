"""Summarize observed mapping recovery and matched one-versus-five comparisons."""
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

from blind_decipher import METHODS, SEEDS
from prepare import write

ROOT=Path("experiments/blind/run-001")
OUT=Path("盲猜结果")


def summary(rows):
    return {"n":len(rows),"metrics":{
        m:{k:float(np.mean([r["metrics"][m][k] for r in rows]))
           for k in ("top1","top5","rank","effective","truth_probability")}
        for m in METHODS}} if rows else {"n":0}


def interval(rows,a,b,metric):
    groups=defaultdict(list)
    for x,y in zip(a,b,strict=True):
        row=rows[x]
        groups[row["gold_uid"]].append(row["metrics"]["gpt_both"][metric]-rows[y]["metrics"]["gpt_both"][metric])
    v=np.array([(sum(g),len(g)) for g in groups.values()])
    rng=np.random.default_rng(51)
    draws=v[rng.integers(len(v),size=(3000,len(v)))].sum(axis=1)
    return np.quantile(draws[:,0]/draws[:,1],[.025,.975]).tolist()


def run():
    OUT.mkdir(exist_ok=True)
    rows={}
    checks=[]
    for seed in SEEDS:
        folder=ROOT/str(seed)
        data=json.loads((folder/"reveal.json").read_text())
        inputs=json.loads((folder/"blind_inputs.json").read_text())
        truth=data["truth"]
        assert len(truth)==len(set(truth.values()))==64
        for observation in inputs:
            assert observation["coded"][observation["position"]]==observation["code"]
            assert not set(observation["coded"])&set(truth.values())
        for item in data["results"]:
            assert truth[item["code"]]==item["gold_uid"]
            used=[inputs[i] for i in item["observations"]]
            assert len({o["component"] for o in used})==len(used)
            assert all(o["code"]==item["code"] for o in used)
            rows[seed,item["cohort"],item["code"],item["budget"]]=item
        for name in ("reveal.json","blind_inputs.json"):
            shutil.copyfile(folder/name,OUT/f"{seed}-{name}")
        for path in (*folder.glob("*.json"),folder/"reverse/gpt.safetensors"):
            checks.append(hashlib.sha256(path.read_bytes()).hexdigest()+"  "+str(path))
    (OUT/"运行文件SHA256.txt").write_text("\n".join(checks)+"\n")
    result={}
    for cohort in ("all","unflagged"):
        matched=[k for k,r in rows.items() if k[1]==cohort and k[-1]==5 and r["n_contexts"]==5]
        first=[(*k[:-1],1) for k in matched]
        result[cohort]={
            "matched_1":summary([rows[k] for k in first]),
            "matched_5":summary([rows[k] for k in matched]),
            "matched_intervals":{m:interval(rows,matched,first,m) for m in ("top1","top5","rank")},
            "coverage_by_seed":{seed:sum(k[0]==seed and k[1]==cohort and k[-1]==5 for k in rows) for seed in SEEDS},
            "up_to_5_by_bin":{b:summary([r for k,r in rows.items() if k[1]==cohort and k[-1]==5 and r["bin"]==b])
                                for b in ("common","mid","rare")},
            "up_to_5":summary([r for k,r in rows.items() if k[1]==cohort and k[-1]==5])}
    write(OUT/"summary.json",result)
    for c,r in result.items():
        print(c,r["coverage_by_seed"],r["matched_intervals"])
        for k in ("matched_1","matched_5","up_to_5"):
            print(k,r[k])


if __name__=="__main__":
    run()
