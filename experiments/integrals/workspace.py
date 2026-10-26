"""Collapsible W&B sections by gamma, with paired train/test curves.

Optional UI dependency: wandb-workspaces==0.4.13. Saved metric histories are
untouched; final-bank results appear as points on the corresponding loss plots.
"""

import argparse
import copy
import json
from pathlib import Path


PANELS = (
    ("answer_nll", "Answer loss", "NLL (nats / answer bit)"),
    ("full_nll", "Full-sequence loss", "NLL (nats / token)"),
    ("non_answer_nll", "Expression and separator loss", "NLL (nats / token)"),
    ("sign_nll", "Sign loss", "NLL (nats / sign bit)"),
    ("exponent_nll", "Exponent loss", "NLL (nats / exponent bit)"),
    ("mantissa_nll", "Mantissa loss", "NLL (nats / mantissa bit)"),
    ("exact_answer_accuracy", "Exact answers", "Fraction correct"),
    ("tolerance_accuracy", "Answers within numerical tolerance", "Fraction correct"),
    ("valid_answer_rate", "Valid answer formats", "Fraction valid"),
    ("decoded_mae", "Numerical error (valid answers only)", "Mean absolute error"),
)


def build_workspace(entity, project, gamma_high=20, start_step=1000, log_y=True):
    import wandb_workspaces.workspaces as ws
    import wandb_workspaces.reports.v2 as wr

    if gamma_high < 1:
        raise ValueError("gamma_high must be positive")
    if start_step < 0:
        raise ValueError("start_step must be nonnegative")
    sections = []
    for gamma in range(1, gamma_high+1):
        panels = []
        for metric, title, ylabel in PANELS:
            keys, marks = [], {}
            for phase in ("eval", "final"):
                for split in ("train", "test"):
                    key = f"{phase}/gamma_{gamma:02d}/{split}/{metric}"
                    keys.append(key)
                    marks[key] = "points" if phase == "final" else ("dashed" if split == "train" else "solid")
            panels.append(wr.LinePlot(
                title=title, x="optimizer_step", y=keys,
                title_x="Optimizer updates", title_y=ylabel,
                # Keep W&B's default run-and-metric legend. A generic Train/Test
                # title hides the condition when several runs share a chart.
                line_marks=marks,
                smoothing_type="none", aggregate=False, max_runs_to_show=4,
                range_y=(0, 1) if metric.endswith(("accuracy", "rate")) else (None, None),
            ))
        sections.append(ws.Section(
            name=f"Gamma {gamma}", panels=panels, is_open=False,
            layout_settings=ws.SectionLayoutSettings(columns=2, rows=5),
        ))
    sections.append(ws.Section(
        name="Run diagnostics", is_open=False,
        layout_settings=ws.SectionLayoutSettings(columns=2, rows=2),
        panels=[
            wr.LinePlot(title="Training objective", x="optimizer_step",
                        y=["train/full_nll", "train/answer_nll"], smoothing_type="none"),
            wr.LinePlot(title="Training gamma", x="optimizer_step", y=["train/gamma"], smoothing_type="none"),
            wr.LinePlot(title="Learning rate", x="optimizer_step", y=["train/learning_rate"], smoothing_type="none"),
            wr.LinePlot(title="Training / evaluation time", x="optimizer_step",
                        y=["time/training_seconds", "time/evaluation_seconds"], smoothing_type="none"),
        ],
    ))
    for section in sections:
        for panel in section.panels:
            panel.range_x = (start_step, None)
            panel.log_x = False
            panel.log_y = log_y
            if log_y:
                # Zero cannot be a logarithmic lower bound (including accuracy).
                # Leave the recorded zeros intact; the chart omits those points.
                panel.range_y = (None, panel.range_y[1])
    return ws.Workspace(entity=entity, project=project,
                        name="Integrals — collapsible gamma sections",
                        sections=sections, auto_generate_panels=False)


def save_workspace(entity, project, output, gamma_high=20, personal_view_id=None,
                   start_step=1000, log_y=True):
    import wandb
    from wandb_workspaces._graphql import execute_graphql
    import wandb_workspaces.workspaces as ws

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    workspace = build_workspace(entity, project, gamma_high, start_step, log_y)
    # Generate the entire layout before changing a live view.
    generated = json.loads(workspace._to_model().spec.model_dump_json(by_alias=True, exclude_none=True))
    api = wandb.Api(timeout=30)
    query = '''query($entity: String!, $project: String!) {
      project(name: $project, entityName: $entity) {
        allViews(viewType: "project-view") { edges { node { id name displayName type spec } } }
      }
    }'''
    before = execute_graphql(api, query, {"entity":entity, "project":project})
    backup = output.with_name(output.stem+"-backup.json")
    if backup.exists():
        raise FileExistsError("Use a new output name to preserve the previous layout backup")
    backup.write_text(json.dumps(before, indent=2)+"\n")
    workspace.save()
    if personal_view_id:
        node = next(e["node"] for e in before["project"]["allViews"]["edges"]
                    if e["node"]["id"] == personal_view_id)
        spec = json.loads(node["spec"])
        section = spec["section"]
        section["panelBankConfig"]["sections"] = copy.deepcopy(generated["section"]["panelBankConfig"]["sections"])
        section.setdefault("workspaceSettings", {})["shouldAutoGeneratePanels"] = False
        if "settings" in section:
            section["settings"]["shouldAutoGeneratePanels"] = False
        # Remove old per-panel placement overrides from the replaced auto panels.
        for key in ("panelConfigOverrides", "panelPlacementOverrides"):
            section["panelBankConfig"].pop(key, None)
        mutation = '''mutation($id: ID, $entity: String, $project: String, $type: String,
                               $name: String, $displayName: String, $spec: String) {
          upsertView(input: {id:$id,entityName:$entity,projectName:$project,type:$type,
                            name:$name,displayName:$displayName,spec:$spec,createdUsing:WANDB_SDK}) {
            view { id name } }
        }'''
        execute_graphql(api, mutation, dict(id=node["id"], entity=entity, project=project,
                        type=node["type"], name=node["name"], displayName=node["displayName"], spec=json.dumps(spec)))
        after = execute_graphql(api, query, {"entity":entity, "project":project})
        updated = next(e["node"] for e in after["project"]["allViews"]["edges"] if e["node"]["id"] == personal_view_id)
        actual = json.loads(updated["spec"])["section"]
        assert actual["workspaceSettings"]["shouldAutoGeneratePanels"] is False
        assert [(s["name"],s["isOpen"]) for s in actual["panelBankConfig"]["sections"]] == [(s.name,False) for s in workspace.sections]
    reloaded = ws.Workspace.from_url(workspace.url)
    assert reloaded.auto_generate_panels is False
    assert all(not s.is_open for s in reloaded.sections)
    assert [s.name for s in reloaded.sections] == [s.name for s in workspace.sections]
    assert all(p.range_x == (start_step, None) and p.log_y == log_y
               for s in reloaded.sections for p in s.panels)
    result = dict(url=workspace.url, entity=entity, project=project, gamma_sections=gamma_high,
                  plots_per_gamma=len(PANELS), all_gamma_sections_collapsed=True,
                  automatic_panels=False, personal_view_updated=personal_view_id,
                  start_step=start_step, logarithmic_y=log_y,
                  final_results="Points on the matching metric plots", backup=str(backup))
    output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default="kanhar")
    parser.add_argument("--project", default="numerical-integrals-curriculum")
    parser.add_argument("--gamma-high", type=int, default=20)
    parser.add_argument("--start-step", type=int, default=1000)
    parser.add_argument("--linear-y", action="store_true", help="Use linear rather than logarithmic metric axes")
    parser.add_argument("--personal-view-id", help="Existing view ID to update, after saving a backup")
    parser.add_argument("--output", default="out/integrals-wandb-layout/gamma-workspace.json")
    args = parser.parse_args()
    save_workspace(args.entity, args.project, args.output, args.gamma_high, args.personal_view_id,
                   args.start_step, not args.linear_y)


if __name__ == "__main__":
    main()
