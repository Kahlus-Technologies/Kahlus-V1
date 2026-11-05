# Autonomic RFS arousal gate

- claim_scope: `autonomic_residual_forecastability_micro_arousal_beyond_cortical_spectral_subject_held_out_nsrr_mesa_shhs_not_gastric_egg_not_coleman_data_not_clinical`
- stop_reason: Autonomic RFS honest negative on MESA under held-out evaluation; do not claim autonomic residual forecastability beyond cortical spectral baseline.
- gate_passed: **False**
- mesa_status: `evaluated`
- shhs_status: `evaluated`
- bootstrap_mode: `smoke`
- epoch_seconds: 30.0

## Estimand

- Y: micro_arousal_within_horizon_epochs
- B: cortical_spectral_plus_cycle_history
- Z: autonomic_hrv_resp_eog_emg_infraslow

## Synthetic known / null

- known primary RFS bits: 0.0211
- null primary RFS bits: -0.0001

## MESA real cohort

- windows: 30945 · subjects: 25
- channels: {'eeg': ['EEG3'], 'ecg': ['EKG'], 'ppg': ['Pleth'], 'eog': ['EOG-L', 'EOG-R'], 'emg': ['EMG'], 'resp': ['Flow', 'Thor', 'Abdo']}
- h=1: positive_events=4379 residual_rfs_bits=0.0061 ci=[-0.0007, 0.0103]

## SHHS dataset-held-out

- windows: 25508 · subjects: 25 · RFS bits: 0.0048
