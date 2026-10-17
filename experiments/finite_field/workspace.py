"""Create a focused W&B workspace: one train/test loss plot per gamma/field.

Requires the optional local dependency wandb-workspaces==0.4.13.
"""

import argparse
import json
from pathlib import Path

from .data import PRIMES


def create_workspace(entity, project, group, output):
    import wandb_workspaces.workspaces as ws
    import wandb_workspaces.reports.v2 as wr
    from wandb_workspaces import expr

    sections = []
    for gamma in range(1, 13):
        panels = []
        for prime in PRIMES:
            prefix = f"eval/gamma_{gamma:02d}/field_{prime:02d}"
            train, test = f"{prefix}/train_loss", f"{prefix}/test_loss"
            panels.append(wr.LinePlot(
                title=f"F{prime} — gamma {gamma}", x="optimizer_step", y=[train, test],
                title_x="Optimizer updates", title_y="Answer cross-entropy (nats)",
                smoothing_type="none", aggregate=False, max_runs_to_show=3,
                line_titles={train: "Train", test: "Test"},
                line_marks={train: "dashed", test: "solid"},
            ))
        sections.append(ws.Section(name=f"Gamma {gamma}", panels=panels, is_open=True,
                                   layout_settings=ws.SectionLayoutSettings(columns=4, rows=2)))
    workspace = ws.Workspace(
        entity=entity, project=project, name="64 layers — gamma 1–12 — train/test loss",
        auto_generate_panels=False, sections=sections,
        runset_settings=ws.RunsetSettings(filters=[expr.Metric("Group") == group]),
    )
    workspace.save()
    result = dict(url=workspace.url, entity=entity, project=project, group=group,
                  sections=12, fields=list(PRIMES), plots=96, automatic_panels=False)
    Path(output).write_text(json.dumps(result, indent=2) + "\n")
    print(workspace.url)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default="kanhar")
    parser.add_argument("--project", default="finite-field-curriculum")
    parser.add_argument("--group", default="gpu-64layer-gamma12")
    parser.add_argument("--output", default="experiments/finite_field/results/gpu_workspace.json")
    args = parser.parse_args()
    create_workspace(args.entity, args.project, args.group, args.output)


if __name__ == "__main__":
    main()
