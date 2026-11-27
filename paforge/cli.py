"""Command line entry points."""
from pathlib import Path
import json
import typer
import yaml
from .config import SimConfig,NoiseConfig,preset
from .pipeline import run_batch
from .datasets import inspect_file,import_measurement
from .io import plot_preview
app=typer.Typer(help="PA-FORGE physics-grounded photoacoustic research prototype")

# Dataset acquisition is opt-in: never download huge third-party archives during install.
datasets_app=typer.Typer(help="Discover and download external photoacoustic research datasets")
app.add_typer(datasets_app,name="datasets")


@datasets_app.command("list")
def datasets_list():
    """List every supported publisher, role, and access mode (offline)."""
    from .dataset_hub import SOURCES
    typer.echo(json.dumps([{'id':s.id,'title':s.title,'provider':s.provider,
                            'role':s.role,'license':s.license_note,'url':s.record,
                            'anatomy':s.anatomy} for s in SOURCES],indent=2))


@datasets_app.command("inspect")
def datasets_inspect(dataset:str):
    """Query upstream metadata; show real filenames, sizes and publisher digests."""
    from .dataset_hub import discover
    try:
        typer.echo(json.dumps(discover(dataset),indent=2))
    except Exception as e:
        typer.echo(f"Publisher discovery failed: {e}",err=True)
        raise typer.Exit(code=1)


@datasets_app.command("plan")
def datasets_plan(task:str=typer.Option("full_research_corpus","--task"),
                 dataset:list[str]=typer.Option([],"--dataset","-d"),
                 profile:str=typer.Option("complete","--profile"),
                 output:Path=typer.Option(Path("data/external"),"--output"),
                 budget:list[str]=typer.Option([],"--budget",help="Per-source whole-asset cap, e.g. synthetic-skin=0.5")):
    """Discover selected publishers and create a storage-aware plan, with individual GB caps."""
    from .dataset_wizard import plan_datasets
    try:
        caps={}
        for token in budget:
            k,v=token.split('=',1);caps[k]=float(v)
        plan=plan_datasets(task,dataset,profile,output,source_budgets_gb=caps)
        typer.echo(json.dumps(plan,indent=2))
    except Exception as exc:
        typer.echo(str(exc),err=True);raise typer.Exit(code=2)


@datasets_app.command("download-plan")
def datasets_download_plan(plan_file:Path=typer.Argument(...),
                           accept_licenses:bool=typer.Option(False,"--accept-licenses")):
    """Execute a previously reviewed dataset wizard plan with resume/checksums."""
    from .dataset_wizard import execute_dataset_plan
    try:
        plan=json.loads(plan_file.read_text(encoding="utf-8"))
        result=execute_dataset_plan(plan,accept_licenses)
        typer.echo(json.dumps(result,indent=2))
    except Exception as exc:
        typer.echo(str(exc),err=True);raise typer.Exit(code=2)


@datasets_app.command("inspect-archives")
def datasets_inspect_archives(root:Path=typer.Option(Path("data/external"),"--root"),
                             output:Path=typer.Option(Path("data/external/archive_inventory.json"),"--output")):
    """Inspect ZIP central directories and data roles without extracting large archives."""
    from .dataset_wizard import inspect_downloaded_archives
    report=inspect_downloaded_archives(root,output)
    typer.echo(json.dumps({'archive_count':report['archive_count'],'output':str(output),
                           'archives':report['archives']},indent=2))


@datasets_app.command("download")
def datasets_download(dataset:list[str]=typer.Option([],"--dataset","-d",help="Repeatable source ID"),
                     all_sources:bool=typer.Option(False,"--all",help="Include every catalog source"),
                     output:Path=typer.Option(Path('data/external'),"--output","-o"),
                     profile:str=typer.Option('starter',"--profile",help="starter or full"),
                     accept_licenses:bool=typer.Option(False,"--accept-licenses",help="Acknowledge upstream license review"),
                     max_file_gb:float=typer.Option(2.,"--max-file-gb"),
                     max_total_gb:float=typer.Option(5.,"--max-total-gb"),
                     unpack:bool=typer.Option(False,"--unpack",help="Optional safe archive unpacking"),
                     dry_run:bool=typer.Option(False,"--dry-run",help="Discover and budget, no downloads")):
    """Download from publisher APIs with resume, checksums and truthful per-source status."""
    from .dataset_hub import run_download
    if not all_sources and not dataset:
        typer.echo("Pass --all or one or more --dataset IDs (see 'paforge datasets list')",err=True)
        raise typer.Exit(code=2)
    try:
        report=run_download(None if all_sources else dataset,output,profile=profile,
                            accept_licenses=accept_licenses,max_file_gb=max_file_gb,
                            max_total_gb=max_total_gb,unpack=unpack,dry_run=dry_run)
    except Exception as e:
        typer.echo(str(e),err=True)
        raise typer.Exit(code=2)
    typer.echo(json.dumps({'status':{k:v['status'] for k,v in report['sources'].items()},
                           'scheduled_gb':round(report['scheduled_bytes']/1e9,3),
                           'report':report.get('report_path'),
                           'details':report['sources']},indent=2))
    if any(x['status'] in ('discovery_error','no_files_downloaded','error') for x in report['sources'].values()):
        raise typer.Exit(code=1)



@app.command()
def demo(output:Path=typer.Option(Path("outputs"),"--output","-o"),
         anatomy:str=typer.Option("forearm","--anatomy"),
         samples:int=typer.Option(1,"--samples"),
         noise:bool=typer.Option(False,"--noise/--no-noise"),
         engine:list[str]=typer.Option(None,"--engine"),
         seed:int=typer.Option(42,"--seed"),
         reconstruction_method:str=typer.Option("das","--reconstruction-method"),
         reconstruction_weights:Path=typer.Option(None,"--reconstruction-weights"),
         grid_size:int=typer.Option(96,"--grid-size"),
         time_steps:int=typer.Option(380,"--time-steps"),
         transducers:int=typer.Option(64,"--transducers"),
         detector_filter:bool=typer.Option(True,"--filter/--no-filter"),
         vascular_pattern:str=typer.Option("branching","--vascular-pattern")) :
    """Run a complete reproducible forward simulation, reconstruction and export."""
    base=preset(anatomy)
    engines=engine if engine else ["gaussian"]
    cfg=SimConfig.model_validate({**base.model_dump(),"samples":samples,"seed":seed,
        "reconstruction_method":reconstruction_method,
        "reconstruction_weights":str(reconstruction_weights) if reconstruction_weights else None,"grid_size":grid_size,
        "time_steps":time_steps,"transducers":transducers,
        "detector_filter":detector_filter,"vascular_pattern":vascular_pattern,
        "noise":{"enabled":noise,"engines":engines}})
    paths=run_batch(cfg,output)
    if cfg.export_truth and cfg.export_images and cfg.reconstruct:
        plot_preview(paths[0])
    typer.echo(json.dumps({"samples":len(paths),"first_sample":str(paths[0]),"noise_enabled":noise},indent=2))


@app.command()
def generate(config:Path, output:Path=typer.Option(Path("outputs"),"--output","-o")):
    """Generate from a YAML config; see examples/config.yaml."""
    cfg=SimConfig.model_validate(yaml.safe_load(config.read_text()))
    paths=run_batch(cfg,output)
    if cfg.export_truth and cfg.export_images and cfg.reconstruct:plot_preview(paths[0])
    typer.echo(f"Completed {len(paths)} samples under {output}")


@app.command("inspect-data")
def inspect_data(path:Path):
    """Inspect uploaded experimental HDF5/MAT/NPZ without assuming array axes."""
    typer.echo(json.dumps(inspect_file(path),indent=2))


@app.command("import-data")
def import_data(path:Path, key:str=typer.Option(...,"--key"),
                output:Path=typer.Option(...,"--output"),
                detector_axis:int=typer.Option(...,"--detector-axis"),
                sampling_rate:float=typer.Option(...,"--sampling-rate-hz"),
                wavelength:float=typer.Option(...,"--wavelength-nm"),
                positions:Path=None):
    """Explicit experimental raw-array import; provenance, axes, units required."""
    dest=import_measurement(path,key,output,detector_axis,sampling_rate,wavelength,positions)
    typer.echo(str(dest))


@app.command("train-demo")
def train_demo(output:Path=typer.Option(Path("examples/demo_cvae.pt"),"--output"),
               epochs:int=typer.Option(1,"--epochs",min=1),
               n:int=typer.Option(32,"--samples",min=8)):
    """Train cVAE on procedural vessels ONLY (not experimental!)."""
    from .learned import create_procedural_training_set,train_cvae
    path=output.with_suffix(".npz")
    create_procedural_training_set(path,n)
    typer.echo(f"Procedural-only vessel masks saved: {path}")
    losses=train_cvae(path,output,epochs=epochs)
    typer.echo(f"Saved {output}. Loss history: {losses}")


@app.command("train-cvae")
def train_real(dataset:Path, weights:Path,epochs:int=10,batch_size:int=16):
    """Train on curated, segmented masks, not on unlabeled raw signals."""
    from .learned import train_cvae
    typer.echo(str(train_cvae(dataset,weights,epochs,batch_size)))



@app.command("benchmark")
def benchmark(output:Path=typer.Option(Path("research_results"),"--output","-o"),
              anatomy:str=typer.Option("vascular","--anatomy"),
              seeds:int=typer.Option(2,"--seeds",min=1,max=50),
              iterations:int=typer.Option(10,"--iterations",min=1,max=100),
              grid:int=typer.Option(64,"--grid",min=48,max=256),
              transducers:int=typer.Option(24,"--transducers",min=8,max=512)):
    """Paired noisy/noiseless synthetic reconstruction baselines. NOT real-data validation."""
    from .research import run_benchmark
    path,summary=run_benchmark(output,anatomy=anatomy,seeds=tuple(7+6*i for i in range(seeds)),
            grid=grid,transducers=transducers,iterations=iterations)
    typer.echo(f"Synthetic-only benchmark: {path}; {len(summary['results'])} measurements")



@app.command("train-vascular-demo")
def train_vascular_demo(weights:Path=typer.Option(Path("models/vascular_synthetic_demo.pt"),"--weights"),
                        samples:int=typer.Option(60,"--samples",min=8),
                        epochs:int=typer.Option(4,"--epochs",min=1)):
    """Train vessel-class CVAE on synthetic masks ONLY (demonstration)."""
    from .vascular_cvae import create_demo_masks,train
    data=weights.with_suffix('.npz')
    create_demo_masks(data,samples)
    history=train(data,weights,epochs=epochs)
    typer.echo(json.dumps({'weights':str(weights),'source':'procedural synthetic only',
        'loss_history':history},indent=2))

@app.command("train-vascular")
def train_vascular(dataset:Path,weights:Path,epochs:int=10,batch_size:int=12):
    """Train four-class vascular CVAE on curated segmentation maps [N,64,64]."""
    from .vascular_cvae import train
    history=train(dataset,weights,epochs=epochs,batch_size=batch_size)
    typer.echo(json.dumps({'weights':str(weights),'loss_history':history},indent=2))



@app.command("make-splits")
def make_splits(metadata_csv:Path,output:Path=typer.Option(Path('splits'),'--output'),
                group_column:str=typer.Option('subject_id','--group-column'),
                seed:int=typer.Option(42,'--seed')):
    """Generate group-disjoint train/validation/test splits from metadata CSV."""
    from .splits import group_split
    manifest=group_split(metadata_csv,output,group_column=group_column,seed=seed)
    typer.echo(json.dumps({'splits':str(output),'groups':manifest['num_independent_groups']},indent=2))



@app.command("extract-acquisition")
def extract_acquisition(path:Path,key:str=typer.Option(...,"--key"),output:Path=typer.Option(...,"--output"),
             detector_axis:int=typer.Option(...,"--detector-axis"),time_axis:int=typer.Option(...,"--time-axis"),
             fixed_axis:list[str]=typer.Option([],"--fixed-axis",help="axis=index, repeat for all other axes"),
             sampling_rate_hz:float=typer.Option(...,"--sampling-rate-hz"),
             wavelength_nm:float=typer.Option(...,"--wavelength-nm"),
             source_id:str=typer.Option(...,"--source-id"),
             group_id:str=typer.Option(...,"--group-id"),
             experimental_attestation:bool=typer.Option(False,"--attest-experimental"),
             positions:Path=None):
    """Extract exactly one genuine raw acquisition from an HDF5 array."""
    from .experimental_import import extract_hdf5_acquisition
    fixed={}
    for item in fixed_axis:
        if '=' not in item:raise typer.BadParameter("--fixed-axis must be axis=index")
        a,i=item.split('=',1);fixed[int(a)]=int(i)
    path=extract_hdf5_acquisition(path,key,output,detector_axis,time_axis,fixed,sampling_rate_hz,
           wavelength_nm,source_id,group_id,experimental_attestation,positions)
    typer.echo(f"Exported original experimental acquisition to {path}")


@app.command("validate-domains")
def validate_domains(manifest:Path=typer.Argument(...,help="CSV manifest of independent real/synthetic acquisitions"),
                     output:Path=typer.Option(Path("experimental_results"),"--output"),
                     extract_quiet_noise:bool=typer.Option(False,"--extract-quiet-noise")):
    """Fit a TRAIN-only spectral match and evaluate a frozen held-out signal distribution."""
    from .experimental import run_experiment
    res=run_experiment(manifest,output,quiet_mode='extract' if extract_quiet_noise else 'none')
    typer.echo(json.dumps({'status':res['status'],'metrics':res['heldout_metrics'],
      'output':str(output)},indent=2))


@app.command("reconstruct-acquisition")
def reconstruct_acquisition(path:Path,output:Path=typer.Option(Path('real_reconstruction'),'--output'),
            method:str=typer.Option('ubp','--method'),fov_m:float=typer.Option(0.04,'--fov-m'),
            grid_size:int=typer.Option(128,'--grid-size'),sound_speed_m_s:float=typer.Option(1500.,'--sound-speed-m-s'),
            positions:Path=None,iterations:int=typer.Option(10,"--iterations"),
            reg:float=typer.Option(.001,"--reg"),
            time_decimate:int=typer.Option(4,"--time-decimate"),weights:Path=None):
    """Qualitative reconstruction from REAL measured signals using real coordinates."""
    from .experimental_reconstruction import reconstruct_experimental
    rec=reconstruct_experimental(path,output,method,fov_m,grid_size,sound_speed_m_s,positions,
                                 iterations=iterations,reg=reg,time_decimate=time_decimate,weights=str(weights) if weights else None)
    typer.echo(json.dumps({'output':str(output),'image_shape':rec.shape,'quantitative':False}))


@app.command("apply-domain-profile")
def apply_domain_profile(synthetic_h5:Path,profile_json:Path,output:Path=typer.Option(...,'--output')):
    """Apply frozen train-only statistical spectral adaptation to synthetic signals."""
    from .domain_adapter import adapt_synthetic
    dest=adapt_synthetic(synthetic_h5,profile_json,output)
    typer.echo(f"Domain-adapted synthetic signals written: {dest} (NOT physical device calibration)")


@app.command("research-sources")
def research_sources():
    """List the small, high-priority public experimental inputs (offline)."""
    from .research_inputs import SMALL_DATASETS
    typer.echo(json.dumps(SMALL_DATASETS,indent=2))


@app.command("inventory-hdf5")
def inventory_hdf5(path:Path,output:Path=None,max_nodes:int=200):
    """Inspect HDF5 arrays, physical metadata and axis candidates without reading full raw data."""
    from .research_inputs import hdf5_inventory
    info=hdf5_inventory(path,max_nodes=max_nodes)
    if output:
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(info,indent=2,default=str))
        typer.echo(f'Inventory written: {output}')
    else: typer.echo(json.dumps(info,indent=2,default=str))


@app.command("benchmark-mismatch")
def benchmark_mismatch(output:Path=typer.Option(Path('research_results/mismatch'),'--output','-o'),
        anatomy:str='vascular',seeds:int=2,grid:int=48,iterations:int=6,
        transducers:int=20,time_steps:int=165,assumed_sound_m_s:float=1630.):
    """Numerical-only FDTD model-mismatch study; not proof of sim-to-real generalization."""
    from .research_v05 import mismatch_benchmark
    result=mismatch_benchmark(output,anatomy=anatomy,seeds=tuple(17+12*i for i in range(seeds)),
               grid=grid,iterations=iterations,transducers=transducers,
               time_steps=time_steps,assumed_sound_m_s=assumed_sound_m_s)
    typer.echo(json.dumps({'status':result['status'],'summary':result['summary'],'folder':str(output)},indent=2))


@app.command('reconstruct-plugin')
def reconstruct_plugin(acquisition_h5:Path, module:str=typer.Option(...,'--module'),
           output:Path=typer.Option(Path('research_results/external_recon'),'--output'),
           grid:int=96, fov_m:float=0.04):
    """Explicitly run an INSTALLED, TRUSTED external Python reconstruction plugin.

    Uses PA-FORGE-native extracted acquisition, NOT an arbitrary vendor HDF5.
    Plugins execute local code; review the code and license before installing.
    """
    from .recon_plugins import run_reconstruction_plugin
    import numpy as np
    import h5py
    from pathlib import Path
    with h5py.File(acquisition_h5,'r') as f:
        if 'raw_data' not in f or 'detector_positions_m' not in f:
            raise typer.BadParameter('Native /raw_data and /detector_positions_m required')
        traces=f['raw_data'][:]
        xyz=f['detector_positions_m'][:]
        fs=float(f.attrs.get('sampling_rate_hz',0))
    img=run_reconstruction_plugin(module,traces,fs,xyz,grid,fov_m)
    output.mkdir(parents=True,exist_ok=True)
    np.save(output/'reconstruction.npy',img)
    meta={'module':module,'input':str(acquisition_h5),'source':'external plugin, user-provided',
          'verified_against_experimental_ground_truth':False,'reconstruction_dimension':2}
    (output/'provenance.json').write_text(json.dumps(meta,indent=2))
    typer.echo(json.dumps({'image_shape':img.shape,'output':str(output),'status':'plugin executed; accuracy not asserted'}))


@app.command("download-openqpai")
def download_openqpai(output:Path=typer.Option(Path('data/openqpai'),'--output'),
                      archive:str=typer.Option('invision','--archive',help='invision/tropus/dis'),
                      unpack:bool=typer.Option(False,'--unpack')):
    """Fetch published experimental archives with original MD5 verification."""
    from .download_data import fetch_openqpai
    typer.echo(json.dumps(fetch_openqpai(output,archive,unpack),indent=2))

@app.command("ui")
def ui(port:int=7860):
    from .ui import build_ui
    build_ui().launch(server_name="127.0.0.1",server_port=port)



@app.command("prepare-openqpai")
def prepare_openqpai_command(
    archive:Path=typer.Argument(...,help="The local InVision ZIP, not a URL"),
    output:Path=typer.Option(Path("models/openqpai"),"--output"),
    absorption_unit:str=typer.Option(...,"--absorption-unit",help="Declare publisher units: 1/m or 1/cm"),
    scattering_unit:str=typer.Option(...,"--scattering-unit",help="Declare publisher units: 1/m or 1/cm"),
    max_groups:int=typer.Option(None,"--max-groups"),
    seed:int=typer.Option(42,"--seed"),
    declared_classes:int=typer.Option(None,"--declared-classes",min=2,max=16)):
    """Prepare group-held-out material maps and TRAIN-only optical sampling profile."""
    from .experimental_learning import prepare_openqpai
    res=prepare_openqpai(archive,output,absorption_unit,scattering_unit,seed,max_groups,declared_classes=declared_classes)
    typer.echo(json.dumps(res,indent=2))


@app.command("prepare-openqpai-multidevice")
def prepare_openqpai_multidevice(
    invision:Path=typer.Option(...,"--invision"),
    dis:Path=typer.Option(...,"--dis"),
    mapping:Path=typer.Option(...,"--mapping"),
    output:Path=typer.Option(Path("models/openqpai_complete"),"--output"),
    tropus:Path=typer.Option(None,"--tropus"),
    svot:Path=typer.Option(None,"--svot"),
    mua_unit:str=typer.Option("1/cm","--mua-unit"),
    musp_unit:str=typer.Option("1/cm","--musp-unit"),
    svot_slices:int=typer.Option(6,"--svot-slices",min=2,max=12)):
    """Prepare InVision/TROPUS/SVOT multi-device corpus with mandatory DIS audit/mapping."""
    from .multidevice_learning import prepare_openqpai_complete
    report=prepare_openqpai_complete(invision,tropus,svot,dis,mapping,output,mua_unit,musp_unit,
                                     seed=42,svot_slices_per_volume=svot_slices)
    typer.echo(json.dumps(report,indent=2))


@app.command("train-material-cvae")
def train_material_cvae_command(
    corpus:Path=typer.Argument(...),profile:Path=typer.Option(...,"--profile"),
    weights:Path=typer.Option(Path("models/material_cvae.pt"),"--weights"),
    epochs:int=typer.Option(20,"--epochs"),batch_size:int=typer.Option(8,"--batch-size"),
    seed:int=typer.Option(42,"--seed"),device:str=typer.Option('auto','--device',help='auto/cpu/cuda/mps'),
    kl_weight:float=typer.Option(1e-4,'--kl-weight',min=0),
    evaluate_test:bool=typer.Option(False,'--evaluate-test/--no-evaluate-test',help='Explicit opt-in: score held-out test masks only after final protocol freeze')):
    """Train a separate material-map CVAE (not vascular) on phantom-group-disjoint data."""
    from .material_cvae import train
    res=train(corpus,profile,weights,epochs,batch_size,seed,device=device,evaluate_test=evaluate_test,kl_weight=kl_weight)
    typer.echo(json.dumps(res,indent=2))


@app.command("generate-informed")
def generate_informed_command(
    model_source:Path=typer.Option(...,"--model-source",help="Trained .pt for material_cvae, or material_corpus.npz for empirical"),
    optical_profile:Path=typer.Option(...,"--optical-profile"),
    engine:str=typer.Option("empirical_material","--engine",help="material_cvae / empirical_material"),
    output:Path=typer.Option(Path("outputs/informed"),"--output"),
    samples:int=typer.Option(2,"--samples",min=1,max=10000),
    seed:int=typer.Option(42,"--seed"),grid_size:int=typer.Option(64,"--grid-size"),
    wavelength_nm:int=typer.Option(700,"--wavelength-nm"),
    fov_m:float=typer.Option(.07,"--fov-m"),
    time_steps:int=typer.Option(380,"--time-steps"),
    transducers:int=typer.Option(48,"--transducers"),
    method:str=typer.Option("ubp","--method"),
    reconstruction_weights:Path=typer.Option(None,"--reconstruction-weights"),
    noise:bool=typer.Option(False,"--noise/--no-noise"),
    noise_std:float=typer.Option(.025,"--noise-std"),
    detector_filter:bool=typer.Option(False,"--filter/--no-filter"),
    iterations:int=typer.Option(8,"--iterations"),
    material_fractions:str=typer.Option("","--material-fractions",help="Optional comma-separated material area proportions"),
    material_solver_padding_factor:int=typer.Option(2,"--material-solver-padding-factor",help="Experimental propagation-domain enlargement: 1 original; 2 zero-pad material p0, keep dx and crop reconstruction")):
    """Generate NEW synthetic raw signals + reconstructed images from train-only experimental phantom priors."""
    from .pipeline import run_batch
    from .io import plot_preview
    if engine not in ('material_cvae','empirical_material'):
        raise typer.BadParameter('engine must be material_cvae or empirical_material')
    cfg=SimConfig.model_validate({**preset('vascular').model_dump(),
        'anatomy':'material_phantom','anatomy_engine':engine,
        'generator_weights':str(model_source),'optical_profile':str(optical_profile),
        'material_composition':[float(z) for z in material_fractions.split(',')] if material_fractions.strip() else None,
        'samples':samples,'seed':seed,'grid_size':grid_size,
        'wavelength_nm':wavelength_nm,'fov_m':fov_m,'time_steps':time_steps,
        'material_solver_padding_factor':material_solver_padding_factor,
        'transducers':transducers,'reconstruction_method':method,
        'reconstruction_weights':str(reconstruction_weights) if reconstruction_weights else None,
        'detector_filter':detector_filter,'reconstruction_iterations':iterations,
        'noise':{'enabled':noise,'engines':['gaussian'],'relative_std':noise_std}})
    out=run_batch(cfg,output)
    if cfg.reconstruct:plot_preview(out[0])
    typer.echo(json.dumps({'output':str(output),'samples':len(out),'first':str(out[0]),
                           'claim':'new EXPERIMENT-INFORMED SYNTHETIC samples, not new experimental measurements'},indent=2))


@app.command("import-ipasc")
def import_ipasc_command(
    path:Path=typer.Argument(...),output:Path=typer.Option(...,"--output"),
    source_id:str=typer.Option(...,"--source-id"),group_id:str=typer.Option(...,"--group-id"),
    origin:str=typer.Option('unknown','--origin',help='experimental/synthetic/unknown, explicitly attested by user'),
    wavelength_index:int=typer.Option(0,"--wavelength-index"),
    frame_index:int=typer.Option(0,"--frame-index")):
    """Normalize a planar IPASC HDF5 using measured geometry and metadata (no synthetic geometry)."""
    from .ipasc_bridge import normalize_ipasc
    dest=normalize_ipasc(path,output,source_id,group_id,origin,wavelength_index,frame_index)
    typer.echo(f'Normalized acquisition: {dest}')


@app.command("extract-invision")
def extract_invision_command(
    archive:Path=typer.Argument(...),
    phantom:str=typer.Option('P.5.10.2','--phantom'),
    wavelength_nm:int=typer.Option(700,'--wavelength-nm'),
    output:Path=typer.Option(Path('data/selected_invision'),'--output'),
    attest_experimental:bool=typer.Option(False,'--attest-experimental',help='Confirm upstream InVision measurements')):
    """Extract one InVision raw IPASC, reference and label and normalize automatically."""
    from zipfile import ZipFile
    from .ipasc_bridge import normalize_ipasc
    import re
    if not re.fullmatch(r'[A-Za-z0-9.\-]+',phantom):
        raise typer.BadParameter('Invalid phantom identifier')
    if not attest_experimental:
        raise typer.BadParameter('Review publisher provenance then pass --attest-experimental')
    candidates=[f'invision/raw/{phantom}_{wavelength_nm}_ipasc.hdf5',
                f'invision/recon/{phantom}_{wavelength_nm}.npy',
                f'invision/labels/{phantom}.npy']
    with ZipFile(archive) as z:
        if any(n not in z.namelist() for n in candidates):
            raise ValueError(f'Files not found in ZIP: {[n for n in candidates if n not in z.namelist()]}')
        output.mkdir(parents=True,exist_ok=True)
        for name in candidates:
            dest=output/name;dest.parent.mkdir(exist_ok=True,parents=True)
            with z.open(name) as src,open(dest,'wb') as target:
                import shutil;shutil.copyfileobj(src,target)
    native=output/'native'/f'{phantom}_{wavelength_nm}.h5'
    normalize_ipasc(output/candidates[0],native,f'OpenQPAIData_InVision_{phantom}_{wavelength_nm}',
                     phantom,'experimental')
    typer.echo(json.dumps({'normalized':str(native),'reference':str(output/candidates[1]),
                           'material_labels':str(output/candidates[2])},indent=2))





@app.command('catalog')
def catalog(source_id:str=typer.Option(None,'--source')):
    """Display dataset roles, source licenses and prohibited cross-domain conversions."""
    from .dataset_registry import all_descriptions,describe
    typer.echo(json.dumps(describe(source_id) if source_id else all_descriptions(),indent=2))


@app.command('inspect-source')
def inspect_source(source:str=typer.Option(...,'--source'),path:Path=typer.Argument(...)):
    """Inspect arbitrary supported publisher archive/numeric arrays without running code."""
    from .dataset_portal import inspect_local_file
    typer.echo(json.dumps(inspect_local_file(source,path),indent=2))


@app.command('preview-source')
def preview_source(source:str=typer.Option(...,'--source'),path:Path=typer.Argument(...),
                   key:str=typer.Option(...,'--key'),member:str=typer.Option(None,'--member'),
                   sample_index:int=typer.Option(0,'--sample-index'),
                   mode:str=typer.Option('image','--mode'),
                   output:Path=typer.Option(Path('outputs/preview'),'--output')):
    """Visualize user-selected 2D array as image or signal without interpreting its meaning."""
    from .dataset_portal import visualize_array
    pics,meta=visualize_array(source,path,key,output,sample_index,member,mode)
    typer.echo(json.dumps({'figures':pics,'manifest':meta},indent=2))


@app.command('extract-skin-pairs')
def skin_pairs(path:Path=typer.Argument(...),output:Path=typer.Option(Path('outputs/skin_pairs'),'--output'),
              samples:int=typer.Option(16,'--samples')):
    """Extract synthetic (not measured) mua/fluence/IP/sinogram references into provenance-linked pairs."""
    from .dataset_portal import extract_skin_pairs
    typer.echo(json.dumps(extract_skin_pairs(path,output,samples),indent=2))


@app.command('import-patato')
def import_patato(path:Path=typer.Argument(...),out:Path=typer.Option(...,'--output'),
                  frame:int=typer.Option(0,'--frame'),
                  wavelength_index:int=typer.Option(0,'--wavelength-index'),
                  group_id:str=typer.Option('unknown','--group-id'),
                  attest_experimental:bool=typer.Option(False,'--attest-experimental')):
    """Normalize a PATATO frame/wavelength with publisher geometry and sampling rate."""
    from .patato_bridge import normalize_patato
    info=normalize_patato(path,out,frame,wavelength_index,
           'experimental' if attest_experimental else 'unknown',attest_experimental,group_id)
    typer.echo(json.dumps(info,indent=2))


@app.command("reconstruction-methods")
def reconstruction_methods():
    """Show every registered reconstruction method and its assumptions."""
    from .reconstruction_engine import method_catalog
    typer.echo(json.dumps(method_catalog(),indent=2))


@app.command("reconstruct-compare")
def reconstruct_compare(input_h5:Path,output:Path=typer.Option(Path('experiments/comparison09'),'--output'),
        methods:str=typer.Option('das,ubp,cf_das,dmas,cg_tikhonov,admm_tv','--methods'),
        fov_m:float=typer.Option(.10,'--fov-m'),grid_size:int=typer.Option(64,'--grid-size'),
        sound_speed_m_s:float=typer.Option(1500.,'--sound-speed-m-s'),
        iterations:int=typer.Option(8,'--iterations'),reg:float=typer.Option(.001,'--reg'),
        time_decimate:int=typer.Option(4,'--time-decimate'),weights:Path=None,truth:Path=None):
    """Reconstruct one measured or synthetic raw acquisition with several methods."""
    from .reconstruction_lab import run_comparison
    report=run_comparison(input_h5,output,methods,grid_size=grid_size,fov_m=fov_m,
                sound_speed_m_s=sound_speed_m_s,iterations=iterations,reg=reg,
                time_decimate=time_decimate,weights=str(weights) if weights else None,truth=truth)
    typer.echo(json.dumps(report,indent=2))


@app.command("train-reconstruction")
def train_reconstruction(paired_root:Path,weights:Path=typer.Option(Path('models/unet_das.pt'),'--weights'),
        architecture:str=typer.Option('unet','--architecture'),epochs:int=typer.Option(5,'--epochs'),
        batch_size:int=typer.Option(4,'--batch-size'),seed:int=typer.Option(42,'--seed'),
        device:str=typer.Option('auto','--device'),max_samples:int=typer.Option(0,'--max-samples')):
    """Supervised training needs synthetic DAS reconstructions paired with p0 truth."""
    from .learned_reconstruction import train_paired
    report=train_paired(paired_root,weights,architecture,epochs,batch_size,seed,device,max_samples)
    typer.echo(json.dumps(report,indent=2))


if __name__ == "__main__":
    app()
