# Google Cloud project

Created on 2026-10-16 for the finite-field curriculum experiment.
On 2026-10-18, both experiment VMs and all their disks were deleted, billing
was disabled, and project deletion was verified as `DELETE_REQUESTED`.
The project was subsequently restored at the user's request; its state was
verified as `ACTIVE` on 2026-10-26. See the [cleanup and restoration record](../CLOUD_CLEANUP.md).
Resource paths and setup commands below are historical.

| Setting | Value |
| --- | --- |
| Project name | Finite Field Curriculum |
| Project ID | `finite-field-curriculum-kanhar` |
| Project number | `820421553694` |
| Billing | Disabled and unlinked on 2026-10-18 |
| Project state | Restored; `ACTIVE` verified on 2026-10-26 |

[Open the project in Google Cloud Console](https://console.cloud.google.com/home/dashboard?project=finite-field-curriculum-kanhar).
The following commands describe the historical project configuration.

```sh
gcloud config set project finite-field-curriculum-kanhar
gcloud projects describe finite-field-curriculum-kanhar
gcloud billing projects describe finite-field-curriculum-kanhar
```

## Training machine

Provisioned on 2026-10-16 after approval of one global GPU and one regional Spot
H100 quota. The completed job is documented in [GPU_RUN.md](GPU_RUN.md), with
final measurements in [GPU_RESULTS.md](GPU_RESULTS.md). The VM and its
persistent disk were deleted on 2026-10-18, including the checkpoints stored
there. The machine details below describe the original allocation.

| Setting | Value |
| --- | --- |
| VM / zone | `finite-field-h100` / `us-central1-a` |
| Machine | Spot `a3-highgpu-1g`, one H100 80 GB, 26 vCPUs, 234 GB RAM |
| Image | `pytorch-2-9-cu129-ubuntu-2404-nvidia-580-v20261001` |
| Runtime | PyTorch 2.9.1+cu129, CUDA 12.9, Python 3.12 |
| Persistent disk | 200 GB balanced boot disk, retained when the VM is deleted |
| Local SSD | Two 375 GB disks; not used for checkpoints |
| Working directory | `/home/kanhar/nanoGPT` |
| Service | `finite-field-training` |
| Monitoring | `finite-field-watch.timer`, every 15 minutes plus trainer exit |
| Shutdown | After verified sweep completion or exhausted recovery; separate 12-hour VM deadline |

The VM has no service account. Only the W&B credential needed by training was
installed privately. GPU compute stops after the job; the persistent disk remains
billable and retains the outputs. Spot interruptions are possible: the checkpoint
and W&B identity allow restart without changing the recipe.

```sh
gcloud compute ssh finite-field-h100 --zone=us-central1-a \
  --project=finite-field-curriculum-kanhar

# On the VM:
cd ~/nanoGPT
systemctl status finite-field-training
tail -n 20 training.log

# Copy all logs, reports, and checkpoints back from the local terminal:
gcloud compute scp --recurse --zone=us-central1-a \
  --project=finite-field-curriculum-kanhar \
  finite-field-h100:/home/kanhar/nanoGPT/out/gpu-64layer-gamma12 \
  out/finite-field/
```

New experiments default to `+`, `-`, and `*`, with division and exponentiation
disabled. Use a fresh output directory and W&B group for the GPU experiment;
the previous CPU results used division and are not a matched baseline for this
changed operator distribution. See [TRACKING.md](TRACKING.md) for W&B setup on
the training machine and [WORKPLAN.md](WORKPLAN.md) for the experimental controls.
