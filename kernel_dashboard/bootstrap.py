"""GPU-host collection, independent of SSH, Colab or other command transports."""
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

root=Path(sys.argv[1]); os.chdir(root)
status={'warnings':[]}
def call(args, output=None, stream=True):
    print('+ '+shlex.join(args),flush=True)
    lines=[]
    with subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1) as process:
        with (Path(output).open('w') if output else open(os.devnull,'w')) as log:
            for line in process.stdout:
                lines.append(line)
                log.write(line); log.flush()
                if stream: print(line,end='',flush=True)
        process.wait()
    proc=subprocess.CompletedProcess(args,process.returncode,''.join(lines))
    print(f'Exit {proc.returncode}'+(f'; saved {output}' if output else ''),flush=True)
    if proc.returncode: status['warnings'].append(f'{args[0]} failed ({proc.returncode}): {proc.stdout[-1200:]}')
    return proc
try:
    # Resolve tools already installed by the agent in the target environment.
    for name, pattern in [('nsys','/opt/nvidia/nsight-systems/*/bin/nsys'),('ncu','/usr/local/cuda*/nsight-compute*/ncu')]:
        if not shutil.which(name):
            import glob
            choices=sorted(glob.glob(pattern))
            if choices: os.environ['PATH']=str(Path(choices[-1]).parent)+':'+os.environ['PATH']
    for tool in ('nsys','ncu'):
        if not shutil.which(tool):
            status['warnings'].append(f'{tool} unavailable; agent must prepare profiling tools on this GPU host')
    entry=[sys.executable,'remote_entry.py','config.json','runner.py','meta.json']
    status['tools']={tool:call([tool,'--version']).stdout if shutil.which(tool) else 'unavailable' for tool in ('nsys','ncu')}
    if shutil.which('nsys'):
        trace=call(['nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--force-overwrite=true','-o','trace',*entry],'nsys.log')
        if Path('trace.nsys-rep').exists():
            call(['nsys','export','--type=sqlite','--force-overwrite=true','--output=trace.sqlite','trace.nsys-rep'],'nsys-export.log')
    if not Path('meta.json').exists():
        call(entry,'runner.log')
    if shutil.which('ncu'):
        query=call(['ncu','--query-metrics','--query-metrics-mode','base'],'available-metrics.txt',stream=False).stdout
        wanted=['dram__bytes_read.sum','dram__bytes_write.sum']
        for prefix in ('f','d'):
            wanted += [f'smsp__sass_thread_inst_executed_op_{prefix}{op}_pred_on.sum' for op in ('add','mul','fma')]
        wanted += [f'sm__ops_path_tensor_src_{kind}_dst_fp32.sum' for kind in ('bf16','fp16','tf32')]
        wanted=[m for m in wanted if m.rsplit('.',1)[0] in query]
        status['requested_metrics']=wanted
        if wanted:
            entry[-1]='ncu-meta.json'
            call(['ncu','--nvtx','--nvtx-include','inference/','--replay-mode','application',
                  '--cache-control','none','--clock-control','none','--metrics',','.join(wanted),
                  '--force-overwrite','-o','counters',*entry],'ncu.log')
            if Path('counters.ncu-rep').exists():
                call(['ncu','--import','counters.ncu-rep','--page','raw','--csv','--print-units','base','--rename-kernels','off'],'counters.csv',stream=False)
    # All Nsight subprocesses have exited; benchmark with a plain Python process.
    call([sys.executable,'remote_entry.py','config.json','runner.py','baseline.json','--benchmark'],'baseline.log')
    if not Path('meta.json').exists(): raise RuntimeError('Runner did not produce metadata; inspect runner/nsys logs')
except Exception as exc:
    status['warnings'].append(str(exc))
finally:
    Path('status.json').write_text(json.dumps(status))
    with tarfile.open('artifacts.tar.gz','w:gz') as archive:
        for p in sorted(root.iterdir()):
            if p.name != 'artifacts.tar.gz' and p.is_file(): archive.add(p,arcname=p.name)
    print(json.dumps({'artifacts':str(root/'artifacts.tar.gz'),'status':status}),flush=True)
