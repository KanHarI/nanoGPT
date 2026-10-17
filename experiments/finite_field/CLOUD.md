# Google Cloud project

Created on 2026-10-16 for the finite-field curriculum experiment.

| Setting | Value |
| --- | --- |
| Project name | Finite Field Curriculum |
| Project ID | `finite-field-curriculum-kanhar` |
| Project number | `820421553694` |
| Billing account | KanHarPersonal |
| Compute Engine API | Enabled |

[Open the project in Google Cloud Console](https://console.cloud.google.com/home/dashboard?project=finite-field-curriculum-kanhar).
The project is the active project in the local `gcloud` configuration.

```sh
gcloud config set project finite-field-curriculum-kanhar
gcloud projects describe finite-field-curriculum-kanhar
gcloud billing projects describe finite-field-curriculum-kanhar
```

## Training machine

Provisioned on 2026-10-16 after approval of one global GPU and one regional Spot
H100 quota. The completed job is documented in [GPU_RUN.md](GPU_RUN.md), with
final measurements in [GPU_RESULTS.md](GPU_RESULTS.md). The VM stopped on
2026-10-17 at 01:40 UTC; its persistent disk retains the checkpoints. SSH and SCP
require the VM to be running.

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
