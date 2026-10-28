"""Produce a disclosure-prudent event-organism aggregate for sharing."""
from pathlib import Path
import pandas as pd

ROOT = Path((__import__("os").environ.get("ARIC_PROJECT_ROOT", "ARIC_PROJECT_ROOT") + '/ARIC_revision_20261009/documents'))
source = pd.read_csv(ROOT / "pathogen_event_summary.csv")
other_label = "Other organism labels"
other = source.loc[source["Organism label"] == other_label, "Events with this organism"]
assert len(other) == 1
rare = source.loc[(source["Organism label"] != other_label) & (source["Events with this organism"] < 11)]
public = source.loc[~source.index.isin(rare.index)].copy()
public.loc[public["Organism label"] == other_label, "Events with this organism"] = int(other.iloc[0] + rare["Events with this organism"].sum())
assert public["Events with this organism"].sum() == source["Events with this organism"].sum()
out = ROOT / "pathogen_event_summary_PUBLIC.csv"
public.to_csv(out, index=False)
print(out)
