from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UpstreamSpec:
    upstream_id: str
    display_name: str
    repo: str
    commit: str | None
    reuse_status: str
    license_status: str
    role: str


def upstream_registry() -> tuple[UpstreamSpec, ...]:
    return (
        UpstreamSpec(
            upstream_id="mamba",
            display_name="Mamba",
            repo="https://github.com/state-spaces/mamba",
            commit="a14b1dff0454a3bc27d9eb31355dc01e4b2490ec",
            reuse_status="permissive",
            license_status="verify upstream license file at pinned commit",
            role="SSM baseline and possible shared dynamics core.",
        ),
        UpstreamSpec(
            upstream_id="ndt3",
            display_name="NDT3",
            repo="https://github.com/joel99/ndt3",
            commit="877f9f5edbcd47f20af6e6c0ba11926da7a22354",
            reuse_status="permissive",
            license_status="verify upstream license file at pinned commit",
            role="Spike/population dynamics baseline.",
        ),
        UpstreamSpec(
            upstream_id="neurostorm",
            display_name="NeuroSTORM",
            repo="https://github.com/CUHK-AIM-Group/NeuroSTORM",
            commit="47028cfdc02b9e1a0cac76d3d9e06057e86846ce",
            reuse_status="mixed",
            license_status="verify code, data, and weight terms separately",
            role="Mamba-based fMRI baseline/reference.",
        ),
        UpstreamSpec(
            upstream_id="neuroformer",
            display_name="Neuroformer",
            repo="https://github.com/a-antoniades/Neuroformer",
            commit="36ed94084628c253098d448048c980d3dea0050d",
            reuse_status="permissive",
            license_status="verify upstream license file at pinned commit",
            role="Spike transformer baseline.",
        ),
        UpstreamSpec(
            upstream_id="neuralbench",
            display_name="NeuralBench / NeuralSet",
            repo="https://github.com/facebookresearch/neuroai",
            commit="30303b368ef2bd3c4524193f9a654c3d89f9d9a3",
            reuse_status="permissive",
            license_status="verify upstream license file at pinned commit",
            role="NeuroAI benchmark/data reference.",
        ),
        UpstreamSpec(
            upstream_id="moabb",
            display_name="MOABB",
            repo="https://github.com/NeuroTechX/moabb",
            commit="68022240adc6efe313638b01807401310176790c",
            reuse_status="permissive",
            license_status="verify upstream license file at pinned commit; datasets vary",
            role="EEG benchmark adapters.",
        ),
        UpstreamSpec(
            upstream_id="cebra",
            display_name="CEBRA",
            repo="https://github.com/AdaptiveMotorControlLab/CEBRA",
            commit="64a06bc7223e688c87549590f63a369a70924ee6",
            reuse_status="mixed",
            license_status="Apache-2.0 for recent versions; track patent notes",
            role="Representation baseline for neural/behavior alignment.",
        ),
        UpstreamSpec(
            upstream_id="braindecode",
            display_name="Braindecode",
            repo="https://github.com/braindecode/braindecode",
            commit="43078cf84482708a3593c3453bef7e1ef9078dfb",
            reuse_status="permissive",
            license_status="verify upstream license file at pinned commit",
            role="EEG specialist baselines and preprocessing.",
        ),
        UpstreamSpec(
            upstream_id="eegpt",
            display_name="EEGPT",
            repo="https://github.com/BINE022/EEGPT",
            commit="a0e0a8fad729e2ecf4eedb3a81548a6e6d48a705",
            reuse_status="permissive",
            license_status="Apache-2.0 code; weights hosted off-repo on figshare, mirror and hash before use",
            role="External pretrained EEG baseline (masked representation learning).",
        ),
        UpstreamSpec(
            upstream_id="labram",
            display_name="LaBraM",
            repo="https://github.com/935963004/LaBraM",
            commit="c431221e6cfd23dbfa9950e0180682fb322b0548",
            reuse_status="mixed",
            license_status="MIT code but Base checkpoint weights committed in-repo; verify code and weight terms separately",
            role="External pretrained EEG baseline (neural-token masked prediction).",
        ),
        UpstreamSpec(
            upstream_id="cbramod",
            display_name="CBraMod",
            repo="https://github.com/wjq-learning/CBraMod",
            commit="b9e961003214326972c567eff390e75b0287e32a",
            reuse_status="permissive",
            license_status="MIT code; weights hosted off-repo on Hugging Face, verify model-card terms",
            role="External pretrained EEG baseline (masked patch reconstruction).",
        ),
        UpstreamSpec(
            upstream_id="biot",
            display_name="BIOT",
            repo="https://github.com/ycq091044/BIOT",
            commit="d138e32634e52ae9fa6ec98ac9c4087b14ca869a",
            reuse_status="mixed",
            license_status="MIT code but checkpoint weights committed in-repo (six-datasets checkpoint slate-excluded); verify code and weight terms separately",
            role="External pretrained EEG baseline (bipolar-montage biosignal transformer).",
        ),
        UpstreamSpec(
            upstream_id="brainlm",
            display_name="BrainLM",
            repo="https://github.com/vandijklab/BrainLM",
            commit=None,
            reuse_status="restricted",
            license_status="CC BY-NC-ND; research-only adapter, no derivative code mixing",
            role="Restricted fMRI reference baseline.",
        ),
        UpstreamSpec(
            upstream_id="spikingjelly",
            display_name="SpikingJelly",
            repo="https://github.com/fangwei123456/spikingjelly",
            commit=None,
            reuse_status="restricted",
            license_status="custom license; isolate before use",
            role="Optional SNN tooling.",
        ),
    )


def permissive_upstreams() -> tuple[UpstreamSpec, ...]:
    return tuple(upstream for upstream in upstream_registry() if upstream.reuse_status == "permissive")


def quarantined_upstreams() -> tuple[UpstreamSpec, ...]:
    return tuple(upstream for upstream in upstream_registry() if upstream.reuse_status != "permissive")
