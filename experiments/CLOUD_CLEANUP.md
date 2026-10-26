# Cloud cleanup — 2026-10-18

At the user's request, the experiment VMs and all attached disks were deleted,
followed by a deletion request for the Google Cloud project. The table below
records the state on 2026-10-18; the project was subsequently restored.

| Resource | Verified outcome |
| --- | --- |
| `finite-field-h100` VM, `us-central1-a` | Deleted |
| `minmax-l4` VM, `us-central1-a` | Deleted |
| Persistent disks with the same names | Both deleted; 200 GB each, 400 GB total |
| Project-wide VM and disk lists | Empty after deletion |
| Disk snapshots, custom images, buckets, reserved IPs, reservations | All lists were empty during inventory |
| Project billing | Disabled; billing account unlinked |
| Project `finite-field-curriculum-kanhar` | Deletion accepted; `DELETE_REQUESTED` verified |

The project display name was **Finite Field Curriculum**, and its project
number was `820421553694`. Google Cloud uses a
[30-day recovery window](https://docs.cloud.google.com/resource-manager/docs/delete-restore-projects)
before permanently deleting a project. The VMs and disks were explicitly
deleted before project shutdown; their checkpoints were deleted with the disks.
The recorded read-back is in
[gcloud_cleanup_2026-10-18.json](minmax/results/gcloud_cleanup_2026-10-18.json).

All three lower-LR gamma-20 runs had finished their full 26,668-update budgets.
Their final reports, configs, source metadata, and W&B histories were downloaded
to `out/minmax-gamma20-low-lr-final/` before cleanup. The validated
[final comparison](minmax/results/gpu_gamma20_low_lr_summary.json) and
[run identities](minmax/results/gpu_gamma20_low_lr_final_runs.json) are retained
in the repository. W&B runs and dashboards remain available, as do previously
saved experiment reports. The cloud cleanup did not remove local files or W&B
data.

VM names, checkpoint paths, SSH commands, and launch instructions in the earlier
experiment documents now describe historical resources. New cloud training
requires new infrastructure.

## Subsequent project restoration

The project was restored at the user's request, while retaining the deletion of
the old VMs and disks. A read-only check on 2026-10-26 confirmed that
`finite-field-curriculum-kanhar` is `ACTIVE`. The original cleanup record above
remains the historical record of deletion, not a claim that the project is still
pending deletion. Compute Engine inventory requests were rejected because billing
is disabled; the resource-deletion evidence remains the 2026-10-18 audit.
The later numerical-integral experiments used the separate
`personal-383909` project; their resources and checkpoint retention are documented
in the [completed integral comparison](integrals/GPU_STAGED_120M.md).
