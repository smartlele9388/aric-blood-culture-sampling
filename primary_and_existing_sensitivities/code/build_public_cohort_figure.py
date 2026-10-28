"""Regenerate the aggregate-only cohort flow without exact rare cells."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

matplotlib.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"], "pdf.fonttype": 42, "svg.fonttype": "none"})

ROOT = Path((__import__("os").environ.get("ARIC_PROJECT_ROOT", "ARIC_PROJECT_ROOT") + '/ARIC_revision_20261009/figures/output'))
ROOT.mkdir(parents=True, exist_ok=True)
with (ROOT / "Figure_1_cohort_flow_public_source.csv").open("w", newline="", encoding="utf-8") as fh:
    writer = csv.writer(fh)
    writer.writerow(("Stage", "Stays_or_count", "Definition"))
    writer.writerows((
        ("Adult ICU stays screened", 94458, "Initial adult screen"),
        ("First ICU stay per hospital admission", 85242, "Retained first stay"),
        ("Stays with qualifying CVC records", 23755, "Qualifying exposure screen"),
        ("No active calendar-day-3 landmark", 1277, "Sequential exclusion"),
        ("Death before landmark", 1226, "Sequential exclusion"),
        ("ICU exit or landmark outside ICU observation", 6746, "Combined sequential exclusion for disclosure"),
        ("Qualifying BSI before landmark", 643, "Sequential exclusion"),
        ("Final ICU stays", 13863, "Calendar-day-3 landmark cohort"),
        ("Distinct patients", 12757, "Final cohort"),
        ("Risk intervals", 60272, "Actual-duration risk periods"),
        ("Risk person-days", 56370.34, "Sum of actual interval duration"),
        ("Noncancelled blood-culture specimens", 19160, "Sampling endpoint"),
        ("First detected line-timed BSI events", 268, "Detected infection endpoint"),
    ))

fig, ax = plt.subplots(figsize=(7.6, 8.0))
fig.patch.set_facecolor("white")
ax.set_xlim(0, 10)
ax.set_ylim(0, 12.5)
ax.axis("off")

def box(x, y, w, h, text, *, fill="white", fontsize=10, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.08", linewidth=1.1, edgecolor="#24445a", facecolor=fill))
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fontsize, weight=weight, color="#152b3a", linespacing=1.3)

def arrow(y_top, y_bottom):
    ax.add_patch(FancyArrowPatch((3.2, y_top), (3.2, y_bottom), arrowstyle="-|>", mutation_scale=13, linewidth=1.1, color="#24445a"))

box(0.8, 11.35, 4.8, 0.78, "94,458 adult ICU stays")
arrow(11.33, 10.75)
box(0.8, 9.88, 4.8, 0.83, "85,242 first ICU stays\nwithin hospital admissions")
arrow(9.85, 9.28)
box(0.8, 8.38, 4.8, 0.85, "23,755 stays with\nqualifying CVC records")

ax.add_patch(FancyArrowPatch((5.65, 8.82), (6.12, 8.82), arrowstyle="-|>", mutation_scale=13, linewidth=1.1, color="#24445a"))
box(6.18, 5.05, 3.15, 6.1,
    "Excluded before landmark: 9,892\n\n"
    "No active day-3 landmark: 1,277\n\n"
    "Death before landmark: 1,226\n\n"
    "ICU exit or landmark outside\nICU observation: 6,746\n\n"
    "Qualifying BSI before\nlandmark: 643",
    fill="#f1f6f9", fontsize=9)

arrow(8.35, 7.75)
box(0.8, 6.75, 4.8, 0.95, "Calendar-day-3 landmark cohort\n13,863 ICU stays; 12,757 patients", fill="#eaf3f8", fontsize=10, weight="bold")
arrow(6.72, 6.15)
box(0.8, 4.85, 4.8, 1.25, "First eligible continuous CVC episode\n60,272 risk intervals\n56,370.34 risk person-days", fontsize=10)
arrow(4.8, 4.18)
box(0.8, 2.8, 4.8, 1.3, "19,160 noncancelled blood-culture specimens\n268 first detected line-timed BSI events", fill="#eaf3f8", fontsize=10)
ax.text(0.8, 1.4, "Sequential exclusions are mutually exclusive. CVC, central venous catheter;\nBSI, bloodstream infection.", fontsize=9, color="#354e5d")

fig.tight_layout(pad=0.45)
fig.savefig(ROOT / "Figure_1_cohort_flow_public.svg", bbox_inches="tight", facecolor="white")
fig.savefig(ROOT / "Figure_1_cohort_flow_public.pdf", bbox_inches="tight", facecolor="white")
fig.savefig(ROOT / "Figure_1_cohort_flow_public.tiff", dpi=600, bbox_inches="tight", facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(ROOT / "Figure_1_cohort_flow_public.png", dpi=600, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(ROOT / "Figure_1_cohort_flow_public.png")
