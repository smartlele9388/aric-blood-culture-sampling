"""Build Figure 5 from final aggregate model outputs; never reads patient rows."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PREVIOUS = ROOT.parent / "ARIC_revision_20261009"
HISTORICAL = ROOT.parent / "ARIC_revision_20260920"
OUT = HERE / "output"
DATA = HERE / "source_data"
QA = HERE / "qa"
for folder in (OUT, DATA, QA):
    folder.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(QA / "matplotlib"))

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

sys.path.insert(0, str(Path.home() / ".codex/skills/nature-figure/scripts"))
from audit_panel_alignment import require_matplotlib_panel_alignment

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.75,
        "ytick.major.size": 0,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "legend.frameon": False,
    }
)

MODEL_SOURCES = [
    {
        "model": "Retrospective component model",
        "timing": "prior_window_plus_discharge_code_count",
        "sampling": HISTORICAL / "association_audit_20261003/corrected_models_final/sampling_components_specimens_final_coefficients.csv",
        "bsi": PREVIOUS / "statistics_audit/r_batch_results/bsi_components_final_r_coefficients.csv",
        "color": "#69747C",
        "marker": "s",
        "offset": -0.24,
        "legend": "With discharge-code count",
    },
    {
        "model": "Diagnosis-free landmark-fixed model",
        "timing": "pre_landmark_fixed",
        "sampling": PREVIOUS / "analysis_core/results/sampling_acute_landmark_only_specimens_coefficients.csv",
        "bsi": PREVIOUS / "statistics_audit/r_batch_results/bsi_acute_landmark_only_r_coefficients.csv",
        "color": "#23608A",
        "marker": "o",
        "offset": 0.0,
        "legend": "No codes; fixed at landmark",
    },
    {
        "model": "Diagnosis-free prior-window updated model",
        "timing": "daily_updated_prior_window",
        "sampling": PREVIOUS / "analysis_core/results/sampling_acute_daily_prior_window_specimens_coefficients.csv",
        "bsi": PREVIOUS / "statistics_audit/r_batch_results/bsi_acute_daily_prior_window_r_coefficients.csv",
        "color": "#B56622",
        "marker": "D",
        "offset": 0.24,
        "legend": "No codes; updated prior window",
    },
]

COMPONENTS = [
    ("Invasive ventilation", "prior_ventilation", "exposure_mechanical_ventilation"),
    ("Documented creatinine rise", "prior_creatinine_rise", "exposure_creatinine_rise_documented"),
    ("Documented lactate ≥2 mmol/L", "prior_high_lactate", "exposure_lactate_high_documented"),
    ("Treatment marker present", "exposure_indication_present", "exposure_indication_present"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exactly_one(table: pd.DataFrame, key: str) -> pd.Series:
    hit = table.loc[table["term"].eq(key)]
    if len(hit) != 1:
        raise ValueError(f"Expected one coefficient for {key}; got {len(hit)}")
    return hit.iloc[0]


def construct_source() -> pd.DataFrame:
    records = []
    source_hashes = {}
    for model in MODEL_SOURCES:
        sampling_path = model["sampling"]
        bsi_path = model["bsi"]
        if not sampling_path.is_file() or not bsi_path.is_file():
            raise FileNotFoundError((sampling_path, bsi_path))
        source_hashes[sampling_path.name] = sha256(sampling_path)
        source_hashes[bsi_path.name] = sha256(bsi_path)
        sampling = pd.read_csv(sampling_path)
        bsi = pd.read_csv(bsi_path)
        for component, sampling_term, bsi_term in COMPONENTS:
            for endpoint, table, term, ratio_column, count_column in (
                ("Sampling", sampling, sampling_term, "rate_ratio", "collections_or_occasions"),
                ("Detected line-timed BSI", bsi, bsi_term, "hazard_ratio", "events"),
            ):
                row = exactly_one(table, term)
                estimate, lower, upper = (float(row[x]) for x in (ratio_column, "ci_lower", "ci_upper"))
                if not (np.isfinite([estimate, lower, upper]).all() and 0 < lower <= estimate <= upper):
                    raise ValueError(f"Invalid {endpoint} interval: {model['model']} / {component}")
                if int(row["intervals"]) != 60272 or int(row["patients"]) != 12757:
                    raise ValueError("Unexpected cohort denominator")
                expected = 19160 if endpoint == "Sampling" else 268
                if int(row[count_column]) != expected:
                    raise ValueError("Unexpected outcome count")
                records.append(
                    {
                        "endpoint": endpoint,
                        "component": component,
                        "model": model["model"],
                        "timing": model["timing"],
                        "source_term": term,
                        "estimate": estimate,
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "p_value": float(row["p_value"]),
                        "risk_intervals": 60272,
                        "patients": 12757,
                        "outcome_count": expected,
                        "source_file": sampling_path.name if endpoint == "Sampling" else bsi_path.name,
                    }
                )
    frame = pd.DataFrame(records)
    if frame.shape[0] != 24 or frame.groupby(["endpoint", "component", "model"]).size().ne(1).any():
        raise ValueError("Figure must retain all 24 acute-component results")
    frame.to_csv(DATA / "Figure_5_source.csv", index=False)
    (QA / "Figure_5_source_provenance.json").write_text(
        json.dumps(
            {
                "model_source_sha256": source_hashes,
                "rows": len(frame),
                "patient_level_data_included": False,
                "confidence_intervals": "Patient-cluster robust 95% confidence intervals; BSI fitted in R survival with Efron ties",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return frame


def draw_forest(ax, frame: pd.DataFrame, endpoint: str, xlim: tuple[float, float], ticks: list[float]):
    if not (0 < xlim[0] < xlim[1]) or np.any(frame["ci_lower"].to_numpy() <= 0):
        raise ValueError("A logarithmic ratio axis requires strictly positive limits and confidence intervals")
    for component_number, (component, _, _) in enumerate(COMPONENTS):
        for model in MODEL_SOURCES:
            record = frame.loc[
                frame["endpoint"].eq(endpoint)
                & frame["component"].eq(component)
                & frame["model"].eq(model["model"])
            ].iloc[0]
            y = component_number + model["offset"]
            ax.errorbar(
                record.estimate,
                y,
                xerr=[[record.estimate - record.ci_lower], [record.ci_upper - record.estimate]],
                fmt=model["marker"],
                color=model["color"],
                ecolor=model["color"],
                markersize=5.2,
                capsize=2.5,
                linewidth=1.05,
                zorder=4,
            )
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([f"{v:g}" for v in ticks]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.axvline(1, color="#75818A", linestyle="--", linewidth=0.9, zorder=1)
    ax.grid(axis="x", color="#E9ECEF", linewidth=0.5, zorder=0)
    for y in (0.5, 1.5, 2.5):
        ax.axhline(y, color="#EBEEF0", linewidth=0.55, zorder=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(3.52, -0.48)
    ax.set_yticks(range(4))
    ax.set_yticklabels([c[0] for c in COMPONENTS])


def main() -> None:
    source = construct_source()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.25), sharey=True)
    fig.subplots_adjust(left=0.325, right=0.975, top=0.73, bottom=0.17, wspace=0.28)
    draw_forest(axes[0], source, "Sampling", (0.85, 1.85), [0.9, 1, 1.2, 1.5, 1.8])
    draw_forest(axes[1], source, "Detected line-timed BSI", (0.68, 2.65), [0.75, 1, 1.5, 2, 2.5])
    axes[0].set_title("a  Blood-culture sampling", loc="left", fontsize=9.5, fontweight="bold", pad=10)
    axes[1].set_title("b  Detected line-timed BSI", loc="left", fontsize=9.5, fontweight="bold", pad=10)
    axes[0].set_xlabel("Adjusted IRR (95% CI)")
    axes[1].set_xlabel("Adjusted HR (95% CI)")
    handles = [
        plt.Line2D([], [], marker=m["marker"], linestyle="None", color=m["color"],
                   markersize=6, label=m["legend"])
        for m in MODEL_SOURCES
    ]
    fig.legend(handles=handles, loc="upper center", ncol=1, bbox_to_anchor=(0.56, 0.995),
               fontsize=8.6, handletextpad=0.5, labelspacing=0.42, borderaxespad=0)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        json_out=QA / "Figure_5.alignment.json",
        overlay_svg=QA / "Figure_5.alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        strict=True,
    )
    stem = OUT / "Figure_5_diagnosis_free_acute_models"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print(f"Figure 5 exported; {len(source)} model-level rows")


if __name__ == "__main__":
    main()
