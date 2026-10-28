"""Check public S1 Poisson covariance CSV against the actual fitted NPZ export."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from audit_baseline_and_period import sha, maxdiff

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--s1-aggregate',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();root=a.s1_aggregate.resolve()
npz=root/'S1_sampling_primary_specimens_covariance.npz'
csv=root/'S1_sampling_primary_specimens_covariance.csv'
bundle=np.load(npz,allow_pickle=True)
terms=list(bundle['terms'])
cov=pd.read_csv(csv,index_col=0)
assert cov.index.tolist()==terms and cov.columns.tolist()==terms
d=maxdiff(cov.to_numpy(),bundle['covariance'])
assert d<1e-15 and np.isfinite(cov.to_numpy()).all()
result={'status':'executed','dimension':list(cov.shape),'term_order_matches_fitted_export':True,'max_abs_CSV_NPZ_difference':d,'finite_covariance':True,'scope':'Export-format verification only; model not refitted','source_sha256':{'fitted_covariance_NPZ':sha(npz),'shared_covariance_CSV':sha(csv),'verification_code':sha(Path(__file__))}}
a.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
