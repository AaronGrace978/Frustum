"""Run independent processes and compare exact provenance/stage/output fingerprints."""
import argparse,json,subprocess,sys,time
from pathlib import Path
from reproducibility import compare_reports

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',type=int,default=3)
 args=p.parse_args()
 if not 2<=args.runs<=5:p.error('Choose 2..5 independent runs')
 here=Path(__file__).parent
 folder=here/'repro_audit'/time.strftime('repeat_%Y%m%d_%H%M%S');folder.mkdir(parents=True,exist_ok=False)
 records=[]
 for i in range(args.runs):
  output=folder/f'run_{i+1}.json'
  command=[sys.executable,str(here/'layer0_block.py'),'--output',str(output)]
  print(f'Running independent process {i+1}/{args.runs}',flush=True)
  result=subprocess.run(command,cwd=here,text=True,capture_output=True,timeout=300)
  (folder/f'run_{i+1}.log').write_text(result.stdout+result.stderr)
  if result.returncode:raise RuntimeError(f'Run {i+1} failed: {result.stderr[-2000:]}')
  records.append(json.loads(output.read_text()))
 checks=[compare_reports(records[0],record) for record in records[1:]]
 baseline=here/'repro_baseline.json'
 baseline_check=compare_reports(json.loads(baseline.read_text()),records[0]) if baseline.exists() else None
 passed=all(r['status']=='PASS' for r in checks) and (baseline_check is None or baseline_check['status']=='PASS')
 report=dict(status='PASS' if passed else 'FAIL',independent_processes=args.runs,comparisons=checks,
  saved_baseline_comparison=baseline_check,evidence_directory=str(folder),
  selected_experts=records[0]['full_prefill_MoE_reference']['selected_experts'],
  output_sha256=records[0]['output_sha256'],
  caveat='Agreement applies to these runs in this environment. The earlier cross-session routing mismatch has no proven historical cause.')
 (here/'reproducibility_results.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
 if not passed:raise SystemExit(1)
if __name__=='__main__':main()
