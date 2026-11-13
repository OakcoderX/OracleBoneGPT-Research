"""Recover randomly relabeled OBIMD classes from held-out contexts."""
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import mlx.core as mx
import numpy as np

from experiment import Corpus, Ngram, SEEDS, mark_duplicates, split
from prepare import CONTEXT, write
from train import GPT, train_gpt

BINS = (("common", 100, 10**9, 24), ("mid", 10, 99, 24), ("rare", 1, 9, 16))
METHODS = ("frequency", "trigram_both", "gpt_left", "gpt_both", "shuffled_context")


def candidates(corpus, seed):
    metadata = json.loads(Path("source/Main-character.json").read_bytes())
    refs = {}
    for uid, m in metadata.items():
        a = m.get("transcription", [])
        if len(a) == 1 and len(a[0]) == 1 and "\u4e00" <= a[0] <= "\u9fff":
            refs[uid] = a[0]
    unique = Counter(refs.values())
    rng = np.random.default_rng(seed+800)
    groups = {}
    for label,lo,hi,n in BINS:
        options = sorted(uid for uid,ref in refs.items()
                         if unique[ref]==1 and lo <= corpus.frequency[uid] <= hi)
        assert len(options) >= n, (label,len(options))
        for uid in rng.choice(options,size=n,replace=False).tolist():
            groups[uid] = label
    labels = sorted(groups)
    aliases = {uid: f"X{code:02d}" for uid,code in zip(labels,rng.permutation(len(labels)),strict=True)}
    return labels, aliases, groups, refs


def reversed_rows(rows):
    return [r | {"tokens":r["tokens"][::-1]} for r in rows]


def load_model(corpus, path):
    model = GPT(len(corpus.input_index),len(corpus.labels))
    model.load_weights(str(path))
    model.eval()
    return model


def neural(model, contexts, corpus, ids):
    results=np.full((len(contexts),len(ids)),-math.log(len(ids)))
    present=[i for i,ctx in enumerate(contexts) if ctx]
    for start in range(0,len(present),64):
        jj=present[start:start+64]
        n=max(len(contexts[i]) for i in jj)
        x=np.full((len(jj),n),corpus.input_index["@pad"],dtype=np.int32)
        for j,i in enumerate(jj):
            ctx=contexts[i]
            x[j,:len(ctx)] = corpus.encode(ctx)
        p=np.array(model(mx.array(x)))
        for j,i in enumerate(jj):
            a=p[j,len(contexts[i])-1,ids].astype(np.float64)
            a-=a.max()
            results[i]=a-np.log(np.exp(a).sum())
    return results


def ngram_scores(model, contexts, corpus, ids):
    values=[]
    for ctx in contexts:
        if not ctx:
            values.append(np.full(len(ids),-math.log(len(ids))))
            continue
        p=model.distribution(tuple(corpus.encode(ctx)),2,8.0)[ids]
        values.append(np.log(p/p.sum()))
    return np.array(values)


def metric(scores, gold):
    order=np.argsort(-scores,kind="stable")
    rank=int(np.flatnonzero(order==gold)[0])+1
    prob=np.exp(scores-scores.max())
    prob/=prob.sum()
    nonzero=prob>0
    return dict(rank=rank,top1=int(rank==1),top5=int(rank<=5),
                effective=float(np.exp(-(prob[nonzero]*np.log(prob[nonzero])).sum())),
                truth_probability=float(prob[gold]), picks=order[:5].tolist())


def run(seed, root):
    directory=root/str(seed);directory.mkdir()
    data,_=split(seed)
    corpus=Corpus(data["train"])
    assert json.loads((Path("runs")/str(seed)/"splits.json").read_text())=={
        k:{r["bone"]:r["component"] for r in rows} for k,rows in data.items()}
    labels,aliases,bins,refs=candidates(corpus,seed)
    ids=np.array([corpus.index[u] for u in labels])
    reverse_train,reverse_val=map(reversed_rows,(data["train"],data["validation"]))
    reverse_path=directory/"reverse";reverse_path.mkdir()
    # The vocabulary is based on the same training rows; reversal preserves counts and IDs.
    model_metadata,_=train_gpt(corpus,reverse_train,reverse_val,[],seed+900,reverse_path)
    fwd=load_model(corpus,Path("runs")/str(seed)/"gpt.safetensors")
    rev=load_model(corpus,reverse_path/"gpt.safetensors")
    mark_duplicates(data["train"],data["test"])
    left,right,observations=[],[],[]
    for r in data["test"]:
        tokens=r["tokens"]
        # Every target-class occurrence becomes a code. ML models receive OOV for
        # noncentral codes; neither their reference value nor their identity is exposed.
        coded=[aliases.get(t,t) for t in tokens]
        concealed=["@oov" if t in aliases else t for t in tokens]
        for i,uid in enumerate(tokens):
            if uid not in aliases:
                continue
            left.append(concealed[:i][-CONTEXT:])
            right.append(concealed[i+1:][::-1][-CONTEXT:])
            observations.append(dict(code=coded[i],bone=r["bone"],component=r["component"],
                index=r["index"],position=i,overlap=r["overlap"],coded=coded))
    gleft=neural(fwd,left,corpus,ids)
    gright=neural(rev,right,corpus,ids)
    trig=(ngram_scores(Ngram(corpus,data["train"]),left,corpus,ids)
          +ngram_scores(Ngram(corpus,reverse_train),right,corpus,ids))/2
    prior=np.log(np.array([corpus.frequency[u] for u in labels]))
    prior-=np.log(np.exp(prior).sum())
    by_code=defaultdict(list)
    for i,o in enumerate(observations):
        by_code[o["code"]].append(i)
    write(directory/"blind_inputs.json",observations)
    # Prediction for each code is independent; no one-to-one assignment is forced.
    rng=np.random.default_rng(seed+1000)
    output=[]
    for cohort in ("all","unflagged"):
        selected={}
        for code,ii in sorted(by_code.items()):
            components={}
            for i in ii:
                row=observations[i]
                if cohort=="all" or row["overlap"]=="unflagged":
                    components.setdefault(row["component"],i)
            if components:
                selected[code]=list(components.values())[:5]
        order=list(selected)
        # Rotate a random ordering so no code gets its own contexts.
        perm=rng.permutation(len(order))
        shuffled={order[a]:selected[order[b]] for a,b in zip(perm,np.roll(perm,1))}
        for uid in labels:
            code=aliases[uid]
            ii=selected.get(code,[])
            if not ii:
                continue
            for budget in (1,5):
                used=ii[:budget]
                false=shuffled[code][:budget]
                both=(gleft+gright)/2
                vectors=(prior,trig[used].sum(axis=0),gleft[used].sum(axis=0),
                         both[used].sum(axis=0),both[false].mean(axis=0)*len(used))
                output.append(dict(code=code,gold_uid=uid,reference=refs[uid],bin=bins[uid],
                    cohort=cohort,budget=budget,n_contexts=len(used),
                    observations=used,metrics={m:metric(v,labels.index(uid))
                    for m,v in zip(METHODS,vectors,strict=True)}))
    write(directory/"reveal.json",dict(candidate_order=[dict(uid=u,reference=refs[u],bin=bins[u]) for u in labels],
          truth={v:u for u,v in aliases.items()},reverse_training=model_metadata,results=output))
    print(seed,len(observations),len(by_code),flush=True)


if __name__=="__main__":
    root=Path("experiments/blind/run-001")
    root.mkdir(parents=True,exist_ok=False)
    for seed in SEEDS:
        run(seed,root)
