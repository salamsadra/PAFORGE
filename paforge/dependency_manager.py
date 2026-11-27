"""Allow-listed local dependency checks and one-click optional installation."""
from __future__ import annotations
from pathlib import Path
import importlib.util, subprocess, sys

MODULES={'numpy':'numpy','scipy':'scipy','h5py':'h5py','matplotlib':'matplotlib',
         'pydantic':'pydantic','typer':'typer','yaml':'PyYAML','gradio':'gradio','torch':'torch','patato':'patato'}


def dependency_status():
    out=[]
    for module,package in MODULES.items():
        found=importlib.util.find_spec(module) is not None
        version=''
        if found:
            try:
                mod=__import__(module);version=str(getattr(mod,'__version__','installed'))
            except Exception:version='installed (version unavailable)'
        out.append({'module':module,'package':package,'installed':found,'version':version})
    try:
        import torch
        out.append({'module':'gpu_runtime','installed':bool(torch.cuda.is_available()),
                    'version':torch.version.cuda or '',
                    'cuda_available':bool(torch.cuda.is_available()),
                    'cuda_device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    'mps_available':bool(hasattr(torch.backends,'mps') and torch.backends.mps.is_available())})
    except ImportError:pass
    return out


def install_selected(component='training_cpu',project_root=None):
    """Install only predefined components; never accept arbitrary pip commands."""
    root=Path(project_root or Path(__file__).resolve().parent.parent).resolve()
    if component not in ('ui','training_cpu','all_cpu','patato'):
        raise ValueError('Choose ui, training_cpu, all_cpu or patato')
    commands=[]
    if component in ('ui','all_cpu'):
        commands.append([sys.executable,'-m','pip','install','--upgrade','gradio>=5'])
    if component in ('training_cpu','all_cpu'):
        # Explicit CPU wheel avoids accidentally pulling a large CUDA runtime.
        commands.append([sys.executable,'-m','pip','install','torch>=2.1','--index-url','https://download.pytorch.org/whl/cpu'])
    if component=='patato':
        commands.append([sys.executable,'-m','pip','install','patato>=0.6.1'])
    logs=[]
    for cmd in commands:
        p=subprocess.run(cmd,cwd=root,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                         timeout=3600,check=False)
        logs.append({'command':cmd,'returncode':p.returncode,'output':p.stdout[-12000:]})
        if p.returncode:
            return {'success':False,'component':component,'logs':logs,
                    'message':'Installation failed. Review the pip output; no arbitrary command was executed.'}
    return {'success':True,'component':component,'logs':logs,
            'message':'Installation completed. Restart PA-FORGE to load newly installed packages.'}
