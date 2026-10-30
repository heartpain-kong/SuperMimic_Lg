import argparse
import json
from pathlib import Path
from supermimic_lg.benchmark import paired_benchmark
from supermimic_lg.simulator import resolve_robot_xml
from supermimic_lg.velocity_physics import TERRAINS


def main():
    p=argparse.ArgumentParser(description="Paired SuperMimic / existing HIM-PPO evaluation on shared velocity physics")
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--him-onnx",type=Path,required=True)
    p.add_argument("--mjlab-root",type=Path,required=True)
    p.add_argument("--robot-xml",type=Path)
    p.add_argument("--terrain",choices=TERRAINS,default="flat")
    p.add_argument("--level",type=int,default=1)
    p.add_argument("--seeds",type=int,nargs="+",default=[0,1,42])
    p.add_argument("--seconds",type=float,default=10.)
    p.add_argument("--backend",choices=("mujoco","mjlab"),default="mujoco")
    p.add_argument("--device",default="cpu")
    p.add_argument("--allow-history-mismatch",action="store_true")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--supermimic-training-manifest",type=Path)
    p.add_argument("--him-training-manifest",type=Path)
    a=p.parse_args()
    report=paired_benchmark(resolve_robot_xml(a.mjlab_root,a.robot_xml),a.mjlab_root,a.checkpoint,a.him_onnx,a.output,
        terrain=a.terrain,level=a.level,seeds=a.seeds,seconds=a.seconds,backend=a.backend,device=a.device,
        allow_history_mismatch=a.allow_history_mismatch)
    manifests={}
    for name,path in (("supermimic",a.supermimic_training_manifest),("him_ppo",a.him_training_manifest)):
        if path:
            manifest=json.loads(path.read_text(encoding="utf-8"))
            if manifest.get("algorithm")!=name or manifest.get("exit_code")!=0 or manifest.get("wall_seconds",-1)<0:
                raise ValueError("invalid/incomplete training manifest")
            manifests[name]=manifest
            report["results"][name]["full_run_training_seconds"]=manifest["wall_seconds"]
    report["training_runs"]=manifests
    report["same_training_hardware"]=(all(manifests["supermimic"].get(k)==manifests["him_ppo"].get(k)
        for k in ('hardware','device','cuda_visible_devices'))) if len(manifests)==2 and all(m.get('device') for m in manifests.values()) else None
    if any(str(m.get('device','')).startswith('cuda') and not m['hardware'].get('gpu_identity_verified') for m in manifests.values()):
        report['same_training_hardware']=None
    report["same_training_task"]=(manifests["supermimic"]["shared_task_contract_sha256"]==manifests["him_ppo"]["shared_task_contract_sha256"]) if len(manifests)==2 and all(m.get("shared_task_contract_sha256") for m in manifests.values()) else None
    a.output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({k:{m:v for m,v in x.items() if m!="episodes"} for k,x in report["results"].items()},indent=2))


if __name__=="__main__": main()
