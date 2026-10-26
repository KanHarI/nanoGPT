"""Create W&B plots, each containing train/test loss at one gamma."""

import argparse
import json
import math
from pathlib import Path


def create_workspace(entity, project, group, output, gamma_high=12):
    import wandb_workspaces.workspaces as ws
    import wandb_workspaces.reports.v2 as wr
    from wandb_workspaces import expr

    if gamma_high < 1:
        raise ValueError("gamma_high must be positive")
    panels = []
    for gamma in range(1, gamma_high + 1):
        train, test = (f"eval/gamma_{gamma:02d}/{split}_loss" for split in ("train", "test"))
        panels.append(wr.LinePlot(
            title=f"Gamma {gamma}", x="optimizer_step", y=[train, test],
            title_x="Optimizer updates", title_y="Answer cross-entropy (nats)",
            smoothing_type="none", aggregate=False, max_runs_to_show=3,
            line_titles={train: "Train", test: "Test"},
            line_marks={train: "dashed", test: "solid"},
        ))
    workspace = ws.Workspace(
        entity=entity, project=project, name=f"Minmax 24 layers gamma 1 to {gamma_high}",
        auto_generate_panels=False,
        sections=[ws.Section(name="Train/test answer loss", panels=panels, is_open=True,
                             layout_settings=ws.SectionLayoutSettings(columns=4, rows=math.ceil(gamma_high / 4)))],
        runset_settings=ws.RunsetSettings(filters=[expr.Metric("Group") == group]),
    )
    workspace.save()
    result = dict(url=workspace.url, entity=entity, project=project, group=group,
                  plots=gamma_high, automatic_panels=False)
    Path(output).write_text(json.dumps(result, indent=2) + "\n")
    print(workspace.url)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default="kanhar")
    parser.add_argument("--project", default="minmax-curriculum")
    parser.add_argument("--group", default="gpu-24layer-gamma12-20m20m")
    parser.add_argument("--output", default="experiments/minmax/results/gpu_workspace.json")
    parser.add_argument("--gamma-high", type=int, default=12)
    args = parser.parse_args()
    create_workspace(args.entity, args.project, args.group, args.output, args.gamma_high)


if __name__ == "__main__":
    main()
