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
STEPS = 900
BATCH = 256
LR = 2e-2

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


def fit_predict(family, width, train_x, train_y, test_x, *, seed=0, steps=STEPS):
    """Identical optimiser and budget for every family -- the comparison is about
    capacity and inductive bias, so nothing else may differ."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))
    net = _build(family, width)
    n_params = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=LR)
    xt = torch.as_tensor(train_x, dtype=torch.float32)
    yt = torch.as_tensor(train_y, dtype=torch.float32)
    rng = np.random.default_rng(seed)
    n = xt.shape[0]
    net.train()
    for _ in range(steps):
        idx = rng.integers(0, n, size=min(BATCH, n))
        opt.zero_grad(set_to_none=True)
        nn.functional.mse_loss(net(xt[idx]), yt[idx]).backward()
        opt.step()
    net.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, test_x.shape[0], 4096):
            chunk = torch.as_tensor(test_x[i:i + 4096], dtype=torch.float32)
            preds.append(net(chunk).numpy())
    return np.concatenate(preds), n_params


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

        alpha, w = tune_ridge(tr_ctx, tr_y, va_ctx, va_y)
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
            pred, n_params = fit_predict(family, width, tr_ctx, tr_y, te_ctx)
            ps = per_subject(corpus, te, h, pred)
            mse = float(np.mean(list(ps.values())))
            d = np.array([ridge_ps[s] - ps[s] for s in te])
            row[f"{family}_{width}"] = {
                "family": family, "width": width, "n_params": n_params,
                "mse": mse, "vs_ridge": paired_stats(d), "per_subject": ps,
                "seconds": round(time.time() - t0, 1),
            }
            print(f"  h={h:>3} {family:>11}_{width:<4} params {n_params:>8,} "
                  f"mse {mse:.4f}  vs_ridge {d.mean():+.4f}  "
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
    probe_grid = [("mlp", 256), ("gru", 256), ("tcn", 128), ("transformer", 128)]
    verdicts, out = [], {}

    for kind in kinds:
        print(f"\n=== fixture: {kind} ===", flush=True)
        corpus = build_fixture(kind, horizons=(1,), n_subjects=40,
                               windows_per_subject=400, seed=SEED)
        tr, va, te = split_subjects(list(corpus), SEED)
        tr_ctx = np.concatenate([corpus[s][0] for s in tr])
        va_ctx = np.concatenate([corpus[s][0] for s in va])
        te_ctx = np.concatenate([corpus[s][0] for s in te])
        tr_y = np.concatenate([corpus[s][1][1] for s in tr])
        va_y = np.concatenate([corpus[s][1][1] for s in va])

        _, w = tune_ridge(tr_ctx, tr_y, va_ctx, va_y)
        ridge_mse = float(np.mean(list(per_subject(corpus, te, 1,
                                                   ridge_predict(w, te_ctx)).values())))
        print(f"  ridge {ridge_mse:.4f}", flush=True)

        per_family = {}
        for family, width in probe_grid:
            pred, n_params = fit_predict(family, width, tr_ctx, tr_y, te_ctx)
            mse = float(np.mean(list(per_subject(corpus, te, 1, pred).values())))
            per_family[f"{family}_{width}"] = mse
            print(f"  {family:>11}_{width:<4} {mse:.4f}  "
                  f"({(ridge_mse - mse) / ridge_mse:+.1%} vs ridge)", flush=True)

        best = min(per_family.values())
        v = check_verdict(kind, ridge_mse, best)
        v["per_family"] = per_family
        verdicts.append(v)
        out[kind] = v
        print(f"  -> expected nonlinear wins={v['expected_nonlinear_wins']}  "
              f"observed={v['observed_nonlinear_wins']}  gap={v['relative_gap']:+.1%}  "
              f"HARNESS {'OK' if v['harness_correct'] else 'WRONG'}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
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
