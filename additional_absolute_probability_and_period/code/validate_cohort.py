#!/usr/bin/env python3
"""Validate the LOCAL one-row-per-stay adapter output; never fits a model.

Standard-library only. Reports contain aggregate counts and row numbers, not IDs.
Usage: python validate_cohort.py cohort.csv --profile S0 --report audit.json
The canonical CSV is restricted MIMIC-derived data and must remain local.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

REQUIRED = {
    "subject_id", "hadm_id", "stay_id", "baseline_group", "age_per_10y",
    "female", "icu_group", "admission_group", "bsi_time_days", "bsi_status",
    "sampling_time_days", "sampling_status", "specimens", "occasions",
    "risk_intervals", "risk_duration_days",
}
SCOPE = "Structural and aggregate validation only; not model or clinical validation"
EXPECTED = {
    "S0": dict(stays=13863, patients=12757, risk_intervals=60272,
               bsi_events=268, specimens=19160, occasions=16086,
               risk_days=56370.34172453703, sampled_stays=5036,
               groups={"low":3046, "moderate":6090, "high":4727},
               exits={0:63, 1:268, 2:1371, 3:9521, 4:2640}),
    "S1": dict(stays=13870, patients=12763, risk_intervals=60333,
               bsi_events=263, specimens=19206, occasions=16127,
               risk_days=56430.68472222222),
}

def numeric(row, name, integer=False):
    value = float(row[name])
    if not math.isfinite(value) or (integer and not value.is_integer()):
        raise ValueError(f"{name}: nonfinite or noninteger")
    return int(value) if integer else value

def audit(rows, profile="S0"):
    errors = []
    if not rows:
        return {"passed":False, "errors":["Empty input"], "profile":profile,"scope":SCOPE}
    absent = sorted(REQUIRED - set(rows[0]))
    if absent:
        return {"passed":False, "errors":["Missing columns: " + ", ".join(absent)], "profile":profile,"scope":SCOPE}
    subjects, stays, admissions = set(), set(), set()
    groups, exits = Counter(), Counter()
    totals = Counter(stays=len(rows), risk_intervals=0, bsi_events=0,
                     specimens=0, occasions=0, sampled_stays=0)
    days = []
    zero_samples = 0
    simultaneous_terminal_samples = 0
    for index, row in enumerate(rows, start=2):
        prefix = f"CSV row {index}: "
        try:
            ids = {k:str(row[k]).strip() for k in ("subject_id", "hadm_id", "stay_id")}
            if any(v.lower() in {"", "na", "nan", "none"} for v in ids.values()):
                raise ValueError("missing identifier")
            if ids["stay_id"] in stays: errors.append(prefix + "duplicate stay")
            if ids["hadm_id"] in admissions: errors.append(prefix + "duplicate admission")
            stays.add(ids["stay_id"]); admissions.add(ids["hadm_id"]); subjects.add(ids["subject_id"])
            group = str(row["baseline_group"]).strip()
            if group not in {"low", "moderate", "high"}: raise ValueError("invalid baseline_group")
            groups[group] += 1
            age = numeric(row, "age_per_10y")
            # Frozen source uses (age - 60) / 10, not age / 10.
            # Source eligibility used anchor_age >= 18. Do not impose a new
            # threshold on the separately reconstructed admission age here.
            if "age" in row and str(row["age"]).strip():
                if not math.isclose(numeric(row,"age"), 10*age+60, abs_tol=1e-7):
                    raise ValueError("age differs from the frozen (age-60)/10 coding")
            if numeric(row, "female", True) not in {0,1}: raise ValueError("female must be 0/1")
            for name in ("icu_group", "admission_group"):
                if str(row[name]).strip().lower() in {"", "na", "nan", "none"}:
                    raise ValueError(f"missing {name}; no silent complete-case deletion")
            bt, st = numeric(row,"bsi_time_days"), numeric(row,"sampling_time_days")
            bs, ss = numeric(row,"bsi_status",True), numeric(row,"sampling_status",True)
            if bs not in range(5) or ss not in range(5): raise ValueError("status outside 0..4")
            if not 0 < bt <= 28 + 1e-9: raise ValueError("BSI follow-up outside (0,28]")
            if not 0 <= st <= bt + 1e-9: raise ValueError("sampling follow-up outside [0,BSI exit]")
            duration = numeric(row,"risk_duration_days")
            if not math.isclose(duration,bt,rel_tol=0,abs_tol=1e-8):
                raise ValueError("risk duration differs from first terminal follow-up")
            if bs == 0 and not math.isclose(bt,28,abs_tol=1e-8):
                raise ValueError("unexpected administrative censoring before day 28")
            spec, occ = numeric(row,"specimens",True), numeric(row,"occasions",True)
            intervals = numeric(row,"risk_intervals",True)
            if not 0 <= occ <= spec or intervals <= 0: raise ValueError("invalid counts")
            if (ss == 1) != (spec > 0) or (spec > 0) != (occ > 0):
                raise ValueError("first-sampling event and collection counts disagree")
            if ss != 1 and (ss != bs or not math.isclose(st,bt,abs_tol=1e-8)):
                raise ValueError("unsampled stay must retain the same terminal exit/time")
            if bs == 1 and ss != 1: raise ValueError("BSI without a qualifying sampling opportunity")
            if ss != 1 and st <= 0: raise ValueError("non-sampling exit at time zero")
            if "baseline_score" in row and str(row["baseline_score"]).strip():
                score = numeric(row,"baseline_score",True)
                if not 0 <= score <= 13: raise ValueError("score outside 0..13")
                implied = "low" if score <= 1 else "moderate" if score <= 3 else "high"
                if group != implied: raise ValueError("baseline score/group mismatch")
            days.append(duration); exits[bs] += 1
            totals.update(risk_intervals=intervals,bsi_events=int(bs==1),
                          specimens=spec,occasions=occ,sampled_stays=int(ss==1))
            zero_samples += int(ss == 1 and st == 0)
            simultaneous_terminal_samples += int(bs==1 and ss==1 and math.isclose(st,bt,abs_tol=1e-9))
        except (ValueError,TypeError,KeyError) as exc:
            errors.append(prefix + str(exc))
    result = dict(totals)
    result.update(patients=len(subjects),risk_days=math.fsum(days),groups=dict(groups),
                  exits=dict(exits),zero_time_first_samples=zero_samples,
                  first_sampling_at_terminal_BSI=simultaneous_terminal_samples)
    for key, expected in EXPECTED.get(profile,{}).items():
        actual = result.get(key)
        ok = (math.isclose(actual,expected,rel_tol=0,abs_tol=1e-6)
              if key == "risk_days" else actual == expected)
        if not ok: errors.append(f"Reference mismatch for {key}: observed {actual}, expected {expected}")
    return {"passed":not errors,"profile":profile,"summary":result,"errors":errors,
            "scope":SCOPE}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv_path",type=Path)
    p.add_argument("--profile",choices=["S0","S1","structural"],default="S0")
    p.add_argument("--report",type=Path,required=True)
    args=p.parse_args()
    reader=gzip.open if args.csv_path.suffix == ".gz" else open
    with reader(args.csv_path,"rt",encoding="utf-8-sig",newline="") as stream:
        rows=list(csv.DictReader(stream))
    report=audit(rows,args.profile)
    digest=hashlib.sha256()
    with args.csv_path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""): digest.update(block)
    report["input_sha256"]=digest.hexdigest()
    args.report.parent.mkdir(parents=True,exist_ok=True)
    with args.report.open("x",encoding="utf-8") as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2)
    print(json.dumps({"passed":report["passed"],"profile":args.profile,
                      "n_errors":len(report["errors"]),"scope":report["scope"]},ensure_ascii=False))
    raise SystemExit(0 if report["passed"] else 2)

if __name__ == "__main__":
    main()
