"""D1 / H1: is there headroom left in single-channel EEG forecasting?

F8 found a GRU beating a tuned ridge out to 160 ms by a modest margin
(0.8298 -> 0.7268 MSE). Before optimising further we need to know whether that
gap is the start of a trend or the end of one. If error stops falling as capacity
and model class increase, the task is at a practical ceiling and the programme
should redirect rather than grind.

Two independent probes, either of which can kill H1:

  A. CAPACITY SATURATION. Sweep model capacity across four architecture families
     (MLP, GRU, dilated TCN, transformer encoder) over roughly two orders of
     magnitude of parameter count, plus a tuned ridge as the linear reference.
     If MSE is flat in log-parameters, adding capacity buys nothing.

  B. PERSONALISATION ORACLE. Refit on a held-out subject's *own* earlier data and
     test on their later data. This is not a deployable model -- it is an upper
     reference. If a model that has seen the individual does no better than the
     cross-subject model, there is no subject-specific structure being missed and
     the remaining error is largely irreducible for this signal.

Pre-registered falsifier (docs/research/program/REGISTER.md, H1): if the largest
model of every family lands within 5% of the tuned ridge at h=16, H1 is dead.

Deliberately reuses the F8 split and loaders so numbers are comparable.

  PYTHONPATH=src python3 scripts/capacity_ceiling.py --out runs/capacity_ceiling
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from amrith_isolated_forecast_check import L, _mse, ridge_predict  # noqa: E402
from forecast_scaleup_paired import (  # noqa: E402
    load_corpus,
    paired_stats,
    split_subjects,
    tune_ridge,
)
from testmode_fixtures import FIXTURES, build_fixture, check_verdict  # noqa: E402

HORIZONS = (1, 16, 64)
SEED = 0
BATCH = 256
MAX_STEPS = 3000
EVAL_EVERY = 100
PATIENCE = 6
#: Ranked on validation with a short probe, then the winner is trained to
#: convergence. A single fixed LR cannot serve all four families -- 2e-2 trains a
#: GRU well and makes a transformer diverge, which is exactly how the first
#: TESTMODE run produced a false "no headroom" verdict.
LR_GRID = (1e-3, 3e-3, 1e-2, 3e-2)
PROBE_STEPS = 400
#: Extended below the shared default (which starts at 0.1) because the fixtures
#: repeatedly selected that bottom edge. An edge selection means the baseline
#: wanted less regularisation than the grid allowed, i.e. it is under-tuned --
#: which would inflate exactly the neural advantage this experiment measures.
RIDGE_ALPHAS_WIDE = (1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 5.0, 20.0, 100.0, 500.0,
                     2000.0, 1e4, 5e4, 2e5, 1e6, 5e6, 2e7)

#: (family, width) -- width spans ~2 orders of magnitude in parameter count so a
#: plateau is visible if one exists.
GRID = [
    ("mlp", 16), ("mlp", 64), ("mlp", 256),
    ("gru", 16), ("gru", 64), ("gru", 256),
    ("tcn", 16), ("tcn", 64), ("tcn", 128),
    ("transformer", 32), ("transformer", 64), ("transformer", 128),
]


def _build(family: str, width: int):
    """One model per (family, width). All consume (batch, L) and emit (batch,)."""
    import torch
    from torch import nn

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(L, width), nn.ReLU(),
                nn.Linear(width, width), nn.ReLU(),
                nn.Linear(width, 1))

        def forward(self, x):
            return self.net(x).squeeze(-1)

    class GRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.GRU(1, width, batch_first=True)
            self.head = nn.Linear(width, 1)

        def forward(self, x):
            out, _ = self.rnn(x.unsqueeze(-1))
            return self.head(out[:, -1, :]).squeeze(-1)

    class TCN(nn.Module):
        """Dilated causal convolutions -- the WaveNet family Pankka et al. used."""

        def __init__(self):
            super().__init__()
            layers, ch_in = [], 1
            for d in (1, 2, 4, 8, 16, 32):
                layers += [nn.Conv1d(ch_in, width, kernel_size=2, dilation=d,
                                     padding=d), nn.ReLU()]
                ch_in = width
            self.body = nn.Sequential(*layers)
            self.head = nn.Linear(width, 1)

        def forward(self, x):
            h = self.body(x.unsqueeze(1))[:, :, :L]
            return self.head(h[:, :, -1]).squeeze(-1)

    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(1, width)
            self.pos = nn.Parameter(torch.randn(1, L, width) * 0.02)
            enc = nn.TransformerEncoderLayer(
                d_model=width, nhead=4, dim_feedforward=2 * width,
                batch_first=True, dropout=0.0)
            self.enc = nn.TransformerEncoder(enc, num_layers=2)
            self.head = nn.Linear(width, 1)

        def forward(self, x):
            h = self.proj(x.unsqueeze(-1)) + self.pos
            return self.head(self.enc(h)[:, -1, :]).squeeze(-1)

    return {"mlp": MLP, "gru": GRU, "tcn": TCN, "transformer": Transformer}[family]()


def _infer(net, x, batch=4096):
    import torch

    preds = []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch):
            preds.append(net(torch.as_tensor(x[i:i + batch], dtype=torch.float32)).numpy())
    return np.concatenate(preds)


def _train(family, width, train_x, train_y, val_x, val_y, *, lr, max_steps,
           seed, early_stop):
    """Train one model, optionally early-stopping on validation MSE."""
    import torch
    from torch import nn

    # Single-threaded on purpose. A GRU over 127 timesteps is sequential in time,
    # so intra-op threading cannot parallelise the recurrence -- it just pays
    # synchronisation once per timestep. Measured on this machine: 284 ms/step at
    # 1 thread vs 538 ms/step at 14. Parallelism belongs across configurations,
    # not inside one.
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    net = _build(family, width)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    xt = torch.as_tensor(train_x, dtype=torch.float32)
    yt = torch.as_tensor(train_y, dtype=torch.float32)
    rng = np.random.default_rng(seed)
    n = xt.shape[0]
    best_val, best_state, bad, used = np.inf, None, 0, 0

    for step in range(1, max_steps + 1):
        net.train()
        idx = rng.integers(0, n, size=min(BATCH, n))
        opt.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(net(xt[idx]), yt[idx])
        if not torch.isfinite(loss):
            break                       # diverged; keep the best state so far
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

        if early_stop and step % EVAL_EVERY == 0:
            net.eval()
            v = _mse(val_y, _infer(net, val_x))
            if v < best_val - 1e-6:
                best_val, bad = v, 0
                best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
                used = step
            else:
                bad += 1
                if bad >= PATIENCE:
                    break

    if early_stop and best_state is not None:
        net.load_state_dict(best_state)
    else:
        net.eval()
        best_val = _mse(val_y, _infer(net, val_x))
        used = max_steps
    net.eval()
    return net, float(best_val), used


def fit_predict(family, width, train_x, train_y, test_x, *, val_x=None, val_y=None,
                seed=0, max_steps=MAX_STEPS):
    """Rank learning rates on validation with a short probe, then train the winner
    to convergence with early stopping.

    Every family gets the same *procedure* and the same budget ceiling. Forcing an
    identical learning rate is not fairness -- it silently cripples whichever
    family that rate does not suit, and produces a plateau that is an artifact of
    optimisation rather than a property of the data.
    """
    n_params = sum(p.numel() for p in _build(family, width).parameters())
    if val_x is None:
        net, _, used = _train(family, width, train_x, train_y, train_x, train_y,
                              lr=1e-2, max_steps=max_steps, seed=seed, early_stop=False)
        return _infer(net, test_x), n_params

    ranked = []
    for lr in LR_GRID:
        _, v, _ = _train(family, width, train_x, train_y, val_x, val_y,
                         lr=lr, max_steps=PROBE_STEPS, seed=seed, early_stop=True)
        ranked.append((v, lr))
    best_lr = min(ranked)[1]
    net, val_mse, used = _train(family, width, train_x, train_y, val_x, val_y,
                                lr=best_lr, max_steps=max_steps, seed=seed,
                                early_stop=True)
    return _infer(net, test_x), n_params, {"lr": best_lr, "steps": used,
                                           "val_mse": val_mse}


def per_subject(corpus, subjects, h, pred_all):
    out, off = {}, 0
    for s in subjects:
        n = corpus[s][0].shape[0]
        out[s] = _mse(corpus[s][1][h], pred_all[off:off + n])
        off += n
    assert off == pred_all.shape[0]
    return out


def probe_a_capacity(corpus, tr, va, te, out_path: Path) -> dict:
    """Does error fall as capacity grows, or flatten?"""
    tr_ctx = np.concatenate([corpus[s][0] for s in tr])
    va_ctx = np.concatenate([corpus[s][0] for s in va])
    te_ctx = np.concatenate([corpus[s][0] for s in te])
    results: dict[str, dict] = {}

    for h in HORIZONS:
        tr_y = np.concatenate([corpus[s][1][h] for s in tr])
        va_y = np.concatenate([corpus[s][1][h] for s in va])
        row: dict[str, dict] = {}

        alpha, w = tune_ridge(tr_ctx, tr_y, va_ctx, va_y, alphas=RIDGE_ALPHAS_WIDE)
        ridge_ps = per_subject(corpus, te, h, ridge_predict(w, te_ctx))
        ridge_mse = float(np.mean(list(ridge_ps.values())))
        row["ridge"] = {"width": None, "n_params": L + 1, "alpha": alpha,
                        "mse": ridge_mse, "per_subject": ridge_ps}
        pers = float(np.mean([_mse(corpus[s][1][h], corpus[s][0][:, -1]) for s in te]))
        row["persistence"] = {"width": None, "n_params": 0, "mse": pers}
        print(f"  h={h:>3} ridge {ridge_mse:.4f} (alpha {alpha:g})  persist {pers:.4f}",
              flush=True)

        for family, width in GRID:
            t0 = time.time()
            pred, n_params, hp = fit_predict(family, width, tr_ctx, tr_y, te_ctx,
                                             val_x=va_ctx, val_y=va_y)
            ps = per_subject(corpus, te, h, pred)
            mse = float(np.mean(list(ps.values())))
            d = np.array([ridge_ps[s] - ps[s] for s in te])
            row[f"{family}_{width}"] = {
                "family": family, "width": width, "n_params": n_params,
                "mse": mse, "vs_ridge": paired_stats(d), "per_subject": ps,
                "chosen": hp, "seconds": round(time.time() - t0, 1),
            }
            print(f"  h={h:>3} {family:>11}_{width:<4} params {n_params:>8,} "
                  f"mse {mse:.4f}  vs_ridge {d.mean():+.4f}  "
                  f"lr={hp['lr']:g} steps={hp['steps']}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
        results[str(h)] = row
        out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


def probe_b_personalisation(corpus, te, out_path: Path) -> dict:
    """Refit on a subject's own earlier windows, test on their later ones.

    Not deployable -- an upper reference. If seeing the individual buys nothing
    over the cross-subject model, there is no subject-specific structure left to
    capture and the residual is largely irreducible.
    """
    results: dict[str, dict] = {}
    for h in HORIZONS:
        gains = []
        for s in te:
            ctx, y = corpus[s][0], corpus[s][1][h]
            n = ctx.shape[0]
            if n < 200:
                continue
            cut = int(0.7 * n)
            # Within-subject ridge, alpha chosen on the subject's own held-out tail.
            mid = int(0.85 * n)
            alpha, w = tune_ridge(ctx[:cut], y[:cut], ctx[cut:mid], y[cut:mid])
            personal = _mse(y[mid:], ridge_predict(w, ctx[mid:]))
            gains.append({"subject": s, "personal_mse": float(personal), "alpha": alpha})
        results[str(h)] = gains
        m = float(np.mean([g["personal_mse"] for g in gains])) if gains else float("nan")
        print(f"  h={h:>3} within-subject ridge mean MSE {m:.4f}  (n={len(gains)})",
              flush=True)
        out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


def run_testmode(args) -> int:
    """Validate the harness against fixtures whose answers we already know.

    A sweep that cannot recover a planted answer cannot be trusted to report an
    honest one. One representative width per family at h=1 -- the fixtures encode
    their dependence in the nearest lags, so a wider grid buys nothing here.
    """
    kinds = list(FIXTURES) if args.testmode == "all" else [args.testmode]
    n_seeds = max(1, args.seeds)
    # Mid-width models, not the largest. The question here is whether the harness
    # can DETECT a 4.5% planted effect, which a width-64 model answers as well as
    # a width-256 one at roughly a quarter of the cost. Capacity itself is what
    # the real sweep varies; this is a sensitivity check.
    probe_grid = [("mlp", 64), ("gru", 64), ("tcn", 64), ("transformer", 64)]
    verdicts, out = [], {}

    for kind in kinds:
      for seed in range(n_seeds):
        tag = kind if n_seeds == 1 else f"{kind}#seed{seed}"
        print(f"\n=== fixture: {tag} ===", flush=True)
        corpus = build_fixture(kind, horizons=(1,), n_subjects=30,
                               windows_per_subject=300, seed=SEED + seed)
        tr, va, te = split_subjects(list(corpus), SEED)
        tr_ctx = np.concatenate([corpus[s][0] for s in tr])
        va_ctx = np.concatenate([corpus[s][0] for s in va])
        te_ctx = np.concatenate([corpus[s][0] for s in te])
        tr_y = np.concatenate([corpus[s][1][1] for s in tr])
        va_y = np.concatenate([corpus[s][1][1] for s in va])

        _, w = tune_ridge(tr_ctx, tr_y, va_ctx, va_y, alphas=RIDGE_ALPHAS_WIDE)
        ridge_mse = float(np.mean(list(per_subject(corpus, te, 1,
                                                   ridge_predict(w, te_ctx)).values())))
        print(f"  ridge {ridge_mse:.4f}", flush=True)

        per_family = {}
        for family, width in probe_grid:
            pred, n_params, hp = fit_predict(family, width, tr_ctx, tr_y, te_ctx,
                                             val_x=va_ctx, val_y=va_y, seed=seed)
            mse = float(np.mean(list(per_subject(corpus, te, 1, pred).values())))
            per_family[f"{family}_{width}"] = {"mse": mse, "chosen": hp}
            print(f"  {family:>11}_{width:<4} {mse:.4f}  "
                  f"({(ridge_mse - mse) / ridge_mse:+.1%} vs ridge)  "
                  f"lr={hp['lr']:g} steps={hp['steps']}", flush=True)

        # Select the family on VALIDATION, then report ITS test MSE. Taking the
        # minimum test MSE over four families is selection on the test set: the
        # max of k noisy estimates is biased upward even when every true effect is
        # zero, so a null fixture yields an apparent win. Measured on `neutral`,
        # where linear is provably optimal, min-over-test gave +1.2% and +2.0% on
        # consecutive seeds -- both in the same direction, which is the signature
        # of bias rather than noise.
        chosen_family = min(per_family.items(),
                            key=lambda kv: kv[1]["chosen"]["val_mse"])[0]
        best = per_family[chosen_family]["mse"]
        v = check_verdict(kind, ridge_mse, best)
        v["chosen_family"] = chosen_family
        v["best_test_mse_if_selected_on_test"] = min(
            x["mse"] for x in per_family.values())
        v["per_family"] = per_family
        v["seed"] = seed
        verdicts.append(v)
        out[tag] = v
        naive = (ridge_mse - v["best_test_mse_if_selected_on_test"]) / ridge_mse
        print(f"  -> val-selected {chosen_family}  gap={v['relative_gap']:+.1%}  "
              f"(selection-on-test would report {naive:+.1%})", flush=True)
        print(f"     expected nonlinear wins={v['expected_nonlinear_wins']}  "
              f"observed={v['observed_nonlinear_wins']}  "
              f"HARNESS {'OK' if v['harness_correct'] else 'WRONG'}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    null_gaps = [v["relative_gap"] for v in verdicts
                 if v["fixture"] in ("neutral", "no_signal")]
    if len(null_gaps) >= 3:
        a = np.array(null_gaps)
        print(f"\nnull-fixture gap distribution over {a.size} runs: "
              f"mean {a.mean():+.4f}  sd {a.std():.4f}  max {a.max():+.4f}  "
              f"p95 {np.percentile(a, 95):+.4f}", flush=True)
        print("  -> a detection threshold must sit ABOVE this, or the harness "
              "reports noise as signal", flush=True)
        out["_null_noise_floor"] = {
            "n_runs": int(a.size), "mean": float(a.mean()), "sd": float(a.std()),
            "max": float(a.max()), "p95": float(np.percentile(a, 95)),
        }
    all_ok = all(v["harness_correct"] for v in verdicts)
    (args.out / "testmode.json").write_text(
        json.dumps({"schema": "kahlus.capacity_testmode.v1",
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "all_fixtures_correct": all_ok, "verdicts": out}, indent=2) + "\n",
        encoding="utf-8")
    print(f"\n{'ALL FIXTURES CORRECT' if all_ok else 'HARNESS FAILED A FIXTURE'} "
          f"({sum(v['harness_correct'] for v in verdicts)}/{len(verdicts)})")
    print(f"wrote {args.out / 'testmode.json'}")
    return 0 if all_ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path,
                    default=Path("/Users/aayu/datasets/kahlus_multidataset_public/sleep-edfx/sleep-cassette"))
    ap.add_argument("--out", type=Path, default=Path("runs/capacity_ceiling"))
    ap.add_argument("--subjects", type=int, default=78)
    ap.add_argument("--max-windows", type=int, default=400)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--probe", choices=["a", "b", "both"], default="both")
    ap.add_argument("--seeds", type=int, default=1,
                    help="repeat each fixture with N seeds; used to measure the "
                         "run-to-run noise floor on the null fixtures")
    ap.add_argument("--testmode", choices=[*FIXTURES, "all"],
                    help="run against planted-signal fixtures with known verdicts "
                         "instead of real data, to validate the harness first")
    args = ap.parse_args()

    if args.testmode:
        return run_testmode(args)

    args.out.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus(args.data, n_subjects=args.subjects, stride=256,
                         max_windows=args.max_windows, workers=args.workers)
    tr, va, te = split_subjects(list(corpus), SEED)
    print(f"split: train {len(tr)}  val {len(va)}  test {len(te)}", flush=True)

    payload = {
        "schema": "kahlus.capacity_ceiling.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "D1", "hypothesis": "H1",
        "falsifier": ("H1 dies if the largest model of every family lands within "
                      "5% of the tuned ridge at h=16"),
        "horizons": list(HORIZONS), "grid": [list(g) for g in GRID],
        "split": {"train": tr, "val": va, "test": te},
    }

    if args.probe in ("a", "both"):
        print("PROBE A -- capacity saturation", flush=True)
        payload["probe_a"] = probe_a_capacity(corpus, tr, va, te,
                                              args.out / "probe_a_partial.json")
    if args.probe in ("b", "both"):
        print("PROBE B -- personalisation oracle", flush=True)
        payload["probe_b"] = probe_b_personalisation(corpus, te,
                                                     args.out / "probe_b_partial.json")

    # Adjudicate the pre-registered falsifier -- mechanically, not by narrative.
    if "probe_a" in payload:
        r16 = payload["probe_a"].get("16", {})
        ridge16 = r16.get("ridge", {}).get("mse")
        if ridge16:
            largest = {}
            for fam in ("mlp", "gru", "tcn", "transformer"):
                cands = [(v["width"], v["mse"]) for k, v in r16.items()
                         if isinstance(v, dict) and v.get("family") == fam]
                if cands:
                    largest[fam] = max(cands)[1]
            within5 = {f: bool(abs(m - ridge16) / ridge16 < 0.05) for f, m in largest.items()}
            payload["falsifier_check"] = {
                "ridge_mse_h16": ridge16, "largest_per_family_mse_h16": largest,
                "within_5pct_of_ridge": within5,
                "H1_falsified": bool(within5 and all(within5.values())),
            }
            print(f"\nfalsifier: {payload['falsifier_check']}", flush=True)

    (args.out / "results.json").write_text(json.dumps(payload, indent=2) + "\n",
                                          encoding="utf-8")
    print(f"wrote {args.out / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
