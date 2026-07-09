"""
Emit Timeloop problem YAMLs (CNN-as-GEMM) for each unique linear-layer shape,
plus a manifest of per-shape multiplicities so postprocess.py can weight totals.
A linear K->N over T tokens is a 1x1 conv: C=K, K_dim=N(out), N=T.  (cnn_layer
dims are R,S,P,Q,C,K,N where K=out, N=batch — see README.)

Two sources of models:
  1. Built-in OPT shapes (MODELS below) — derived from architecture params.
  2. models/*.json exported by the Colab experiment cell (FP_Quant.ipynb) — the
     ACTUAL per-layer (K,N) dims of ANY network (Llama GQA / gated MLP / OPT).
     Drop a downloaded <model>_hwsim.json into models/ and re-run this.
"""
import os, json, yaml

T = 2048  # tokens (batch N)
MODELS = {
    "opt-125m": dict(d=768,  ff=3072,  n_layers=12, vocab=50272),
    "opt-1.3b": dict(d=2048, ff=8192,  n_layers=24, vocab=50272),
}
HERE = os.path.dirname(os.path.abspath(__file__))


def opt_shapes(d, ff, n_layers, vocab):
    # (shape_name, C=in, K_out, count over the whole model)
    return [
        ("attn_dxd", d,  d,  4 * n_layers),   # q,k,v,out
        ("fc1_dxff", d,  ff, 1 * n_layers),
        ("fc2_ffxd", ff, d,  1 * n_layers),
        ("lm_head",  d,  vocab, 1),
    ]


def json_shapes(path):
    """(model_key, [(shape_name, C, K_out, count), ...]) from a Colab export."""
    d = json.load(open(path))
    key = d["model"].split("/")[-1]
    shapes = [(f"K{s['K']}_N{s['N']}", int(s["K"]), int(s["N"]), int(s["count"]))
              for s in d["shapes"]]
    return key, shapes


def prob_yaml(C, K_out):
    return {"problem": {"version": 0.3, "instance": {
        "C": C, "K": K_out, "N": T,
        "R": 1, "S": 1, "P": 1, "Q": 1}}}


def collect_models():
    models = {k: opt_shapes(**cfg) for k, cfg in MODELS.items()}
    mdir = os.path.join(HERE, "models")
    if os.path.isdir(mdir):
        for fn in sorted(os.listdir(mdir)):
            if fn.endswith(".json"):
                key, shapes = json_shapes(os.path.join(mdir, fn))
                models[key] = shapes
                print(f"  ingested models/{fn} -> {key} ({len(shapes)} unique shapes)")
    return models


def main():
    outdir = os.path.join(HERE, "prob")
    os.makedirs(outdir, exist_ok=True)
    manifest = {}
    nfiles = 0
    for mkey, shapes in collect_models().items():
        manifest[mkey] = []
        for name, C, K_out, count in shapes:
            fname = f"{mkey}__{name}.yaml"
            with open(os.path.join(outdir, fname), "w") as f:
                yaml.safe_dump(prob_yaml(C, K_out), f, sort_keys=False)
            manifest[mkey].append({"shape": name, "file": fname,
                                   "C": C, "K": K_out, "N": T, "count": count})
            nfiles += 1
    with open(os.path.join(HERE, "manifest.yaml"), "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)
    print(f"wrote {nfiles} problem files + manifest.yaml ({len(manifest)} models)")


if __name__ == "__main__":
    main()
