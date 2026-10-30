"""Record full-process wall time and hardware for either training command."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from supermimic_lg.benchmark import hardware, digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--algorithm",choices=("supermimic","him_ppo"),required=True)
    p.add_argument("--task-config",type=Path,help="saved actual training config, preserved as evidence")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("command",nargs=argparse.REMAINDER)
    a=p.parse_args()
    command=a.command[1:] if a.command[:1]==["--"] else a.command
    if not command: p.error("provide the training command after --")
    if a.output.exists(): raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    manifest=dict(algorithm=a.algorithm,command=command,hardware=hardware(),
        task_config=str(a.task_config.resolve()) if a.task_config else None,
        task_config_sha256=None,
        timing="full subprocess wall time including initialization/compilation/evaluation performed inside the command")
    manifest['device']=next((command[i+1] for i,x in enumerate(command[:-1]) if x=='--device'),
                            next((x.split('=',1)[1] for x in command if x.startswith('--device=')),None))
    manifest['cuda_visible_devices']=os.environ.get('CUDA_VISIBLE_DEVICES')
    flags={"creationflags":subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.CREATE_NO_WINDOW} if os.name=="nt" else {"start_new_session":True}
    started=time.monotonic()
    with a.output.with_suffix(".log").open("w",encoding="utf-8") as stream:
        process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,**flags)
        print(f"Training PID {process.pid}; log: {a.output.with_suffix('.log')}",flush=True)
        try:
            process.wait()
        except KeyboardInterrupt:
            if os.name=="nt":
                subprocess.run(["taskkill","/PID",str(process.pid),"/T","/F"],check=False,capture_output=True)
            else:
                os.killpg(process.pid,signal.SIGINT)
                try: process.wait(timeout=15)
                except subprocess.TimeoutExpired: os.killpg(process.pid,signal.SIGKILL)
            process.wait()
        finally:
            manifest.update(wall_seconds=time.monotonic()-started,exit_code=process.returncode)
            if a.task_config and a.task_config.is_file():
                manifest['task_config_sha256']=digest(a.task_config)
                try:
                    config=json.loads(a.task_config.read_text(encoding='utf-8'))
                    manifest['shared_task_contract_sha256']=config.get('shared_task_contract_sha256')
                except (ValueError,AttributeError):pass
            a.output.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    raise SystemExit(process.returncode or 0)


if __name__=="__main__": main()
