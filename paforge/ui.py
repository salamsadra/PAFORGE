"""Interactive Gradio control panel. Runs locally, no cloud data transfer."""
import tempfile
import zipfile
from pathlib import Path
from .config import preset,SimConfig
from .pipeline import run_batch
from .io import plot_preview
from .reconstruction_engine import METHODS as RECON_METHODS, method_catalog


def _bundle(folder, only_samples=None):
    folder=Path(folder)
    archive=folder/"pa_forge_outputs.zip"
    if only_samples is None:
        # Legacy callers use a dedicated temporary or workflow root.
        files=folder.rglob("*")
    else:
        # Experiment-informed UI: NEVER bundle unrelated files from a user-picked
        # persistent directory (which could contain unrelated or held-out data).
        selected=[Path(p) for p in only_samples]
        if not selected or any(not p.is_dir() or p.parent.resolve()!=folder.resolve()
                               or p.is_symlink() for p in selected):
            raise ValueError('Bundle inputs must be existing direct sample folders')
        if archive.exists():
            raise FileExistsError('Refusing to overwrite existing dataset ZIP; choose a new output directory')
        files=(f for sample in selected for f in sample.rglob("*"))
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for f in files:
            if f==archive or not f.is_file():continue
            if f.is_symlink() or not f.resolve().is_relative_to(folder.resolve()):
                raise ValueError(f'Unsafe linked file in output: {f}')
            z.write(f,f.relative_to(folder))
    return str(archive)


def generate_ui(anatomy,samples,seed,noise_enabled,noise_engines,noise_std,empirical_path,
                wavelength,transducers,geometry,center_mhz,bandwidth_mhz,
                sound_var,scattering,laser_jitter,dropout,grid,time_steps,
                vascular_pattern,vascular_branches,vascular_density,
                recon_method,recon_iterations,filter_enabled,
                illumination,anatomy_engine,generator_weights,reconstruction_weights=""):
    base=preset(anatomy)
    config={**base.model_dump(),"samples":int(samples),"seed":int(seed),
            "grid_size":int(grid),"time_steps":int(time_steps),
            "vascular_pattern":vascular_pattern,"vascular_branch_levels":int(vascular_branches),
            "vascular_density":vascular_density,"reconstruction_method":recon_method,
            "illumination_model":illumination,"anatomy_engine":anatomy_engine,
            "generator_weights":generator_weights.strip() or None,
            "reconstruction_weights":str(reconstruction_weights or "").strip() or None,
            "reconstruction_iterations":int(recon_iterations),"detector_filter":bool(filter_enabled),
            "wavelength_nm":int(wavelength),"transducers":int(transducers),
            "geometry":geometry,"center_frequency_hz":center_mhz*1e6,
            "bandwidth_hz":bandwidth_mhz*1e6,
            "sound_speed_variation_fraction":sound_var,"scattering_scale":scattering,
            "pulse_energy_jitter_fraction":laser_jitter,
            "channel_dropout_probability":dropout,
            "noise":{"enabled":bool(noise_enabled),"engines":noise_engines,
                     "relative_std":noise_std,"empirical_path":empirical_path.strip() or None}}
    cfg=SimConfig.model_validate(config)
    output=Path(tempfile.mkdtemp(prefix="paforge_"))
    results=run_batch(cfg,output)
    overview=plot_preview(results[0])
    archive=_bundle(output)
    import json
    return str(overview),archive,json.dumps({"config":cfg.model_dump(mode="json"),"output_dir":str(output)},indent=2)



def _load_reference(reference_path):
    import numpy as np
    if not reference_path or not str(reference_path).strip():
        return None
    p=Path(str(reference_path).strip().strip('"'))
    if not p.is_file() or p.suffix.lower()!='.npy':
        raise ValueError('Reference path must point to an existing .npy file')
    a=np.load(p,allow_pickle=False)
    if a.ndim!=2:
        raise ValueError('Reference must be a 2D NumPy image')
    return a


def _preview_panel(reconstruction,reference=None,rotation=0,flip_lr=False,flip_ud=False):
    import numpy as np
    from matplotlib import colormaps
    from PIL import Image
    def picture(data,rotate=False):
        if data is None:return None
        a=np.asarray(data,dtype=float)
        if rotate:
            a=np.rot90(a,int(rotation)//90)
            if flip_lr:a=np.fliplr(a)
            if flip_ud:a=np.flipud(a)
        a=np.nan_to_num(a,nan=0,posinf=0,neginf=0)
        lo,hi=np.percentile(a,[1,99.5])
        scaled=np.clip((a-lo)/max(float(hi-lo),1e-12),0,1)
        return (colormaps['magma'](scaled)[:,:,:3]*255).astype(np.uint8)
    return picture(reconstruction,True),picture(reference)


def _experimental_dashboard_reconstruct(local_path,uploaded_file,reference_path,method,fov_mm,grid_size,speed,rotation,flip_lr,flip_ud,iterations=10,reg=0.001,time_decimate=4,weights=""):
    import h5py
    import numpy as np
    import json
    import tempfile
    import zipfile
    from .experimental_reconstruction import reconstruct_experimental
    from .research_inputs import hdf5_inventory
    src=str(local_path or '').strip().strip('"')
    if not src and uploaded_file:
        src=str(uploaded_file)
    if not src or not Path(src).is_file():
        raise ValueError('Provide an existing local normalized PA-FORGE HDF5 path or upload that HDF5 file')
    with h5py.File(src,'r') as f:
        if 'raw_data' not in f or 'detector_positions_m' not in f:
            raise ValueError('This panel expects a normalized acquisition with /raw_data and /detector_positions_m. Use extract-acquisition first; raw IPASC is not directly interchangeable.')
        shape=f['raw_data'].shape
        pos=f['detector_positions_m'][:]
        fs=float(f.attrs.get('sampling_rate_hz',0))
        synthetic=bool(f.attrs.get('synthetic',False))
    if shape[0]!=pos.shape[0] or len(shape)!=2:raise ValueError('Detector count and signals shape mismatch')
    if np.ptp(pos[:,2])>0.002:raise ValueError('Significant 3D spread; this panel only supports justified planar 2D data')
    output=Path(tempfile.mkdtemp(prefix='paforge_real_dashboard_'))
    rec=reconstruct_experimental(src,output,method=method,fov_m=float(fov_mm)/1000,grid_size=int(grid_size),sound_speed_m_s=float(speed),iterations=int(iterations),reg=float(reg),time_decimate=int(time_decimate),weights=str(weights or "").strip() or None)
    ref=_load_reference(reference_path)
    left,right=_preview_panel(rec,ref,rotation,flip_lr,flip_ud)
    description={'source':src,'signals_shape':list(shape),'sampling_rate_hz':fs,'detector_count':int(shape[0]),'synthetic':synthetic,'reconstruction_method':method,'fov_mm':float(fov_mm),'grid_size':int(grid_size),'assumed_sound_speed_m_s':float(speed),'iterations':int(iterations),'regularization':float(reg),'time_decimate':int(time_decimate),'reference_shape':list(ref.shape) if ref is not None else None,'display_rotation_ccw_deg':int(rotation),'display_flip_lr':bool(flip_lr),'display_flip_ud':bool(flip_ud),'warning':'Qualitative 2D reconstruction only. Visual transformations apply to preview only; physical registration and calibration are not established.'}
    (output/'dashboard_run.json').write_text(json.dumps(description,indent=2),encoding='utf-8')
    archive=_bundle(output)
    return left,right,json.dumps(description,indent=2),archive,rec,ref


def _display_orient(rec,ref,rotation,flip_lr,flip_ud):
    if rec is None:return None,None
    return _preview_panel(rec,ref,rotation,flip_lr,flip_ud)


def _summarize_acquisition(path,uploaded):
    import h5py, json, numpy as np
    source=str(path or '').strip().strip('"') or str(uploaded or '')
    if not source or not Path(source).is_file():return 'Select an HDF5 acquisition to inspect.'
    with h5py.File(source,'r') as f:
        info={'file':source,'datasets':{k:list(f[k].shape) for k in ('raw_data','detector_positions_m') if k in f},'attributes':{k:str(v) for k,v in f.attrs.items()}}
        if 'detector_positions_m' in f:
            p=f['detector_positions_m'][:]
            info['coordinate_spread_m']=(np.ptp(p,axis=0)).tolist()
    return json.dumps(info,indent=2)


def _ui_download(sources,profile,max_file,max_total,accept,folder):
    import json
    from .dataset_hub import run_download
    if not sources:raise ValueError('Choose one or more publisher sources')
    if not accept:raise ValueError('Review source-specific upstream licenses, then tick acceptance')
    report=run_download(sources,folder,profile=profile,accept_licenses=True,
                        max_file_gb=float(max_file),max_total_gb=float(max_total),dry_run=False)
    return json.dumps({'status':{k:v['status'] for k,v in report['sources'].items()},
                       'report':report.get('report_path'),
                       'selected_file_count':sum(len(v.get('assets',[])) for v in report['sources'].values())},indent=2)


def _ui_inspect_source(local_path,uploaded_file):
    from .research_inputs import hdf5_inventory
    import json
    src=str(local_path or '').strip().strip('"') or str(uploaded_file or '')
    if not src:raise ValueError('Provide local HDF5 file path or upload')
    report=hdf5_inventory(src,max_nodes=18)
    return json.dumps(report,indent=2,default=str)


def _ui_import_ipasc(local_path,uploaded_file,out_path,source_id,group_id,origin,confirm):
    from .ipasc_bridge import normalize_ipasc
    src=str(local_path or '').strip().strip('"') or str(uploaded_file or '')
    if not src:raise ValueError('Provide a raw IPASC HDF5')
    if origin=='experimental' and not confirm:
        raise ValueError('Please confirm that publisher documents establish experimental acquisition provenance')
    dest=normalize_ipasc(src,out_path,source_id,group_id,origin)
    return str(dest),f'Normalized 2D acquisition saved: {dest}. Metadata-derived detector plane; no fabricated geometry.'


def _ui_select_invision(archive,phantom,wavelength,outdir,confirm):
    from zipfile import ZipFile
    import shutil,re,json
    from .ipasc_bridge import normalize_ipasc
    if not confirm:raise ValueError('Review and attest experimental InVision source provenance')
    if not re.fullmatch(r'[A-Za-z0-9.-]+',str(phantom)):raise ValueError('Invalid phantom name')
    archive=Path(archive);out=Path(outdir)
    stems=[f'invision/raw/{phantom}_{int(wavelength)}_ipasc.hdf5',
           f'invision/recon/{phantom}_{int(wavelength)}.npy',
           f'invision/labels/{phantom}.npy']
    with ZipFile(archive) as z:
        if any(x not in z.namelist() for x in stems):raise ValueError('Phantom/wavelength filenames unavailable in ZIP')
        for name in stems:
            dest=out/name;dest.parent.mkdir(parents=True,exist_ok=True)
            with z.open(name) as source,open(dest,'wb') as target:shutil.copyfileobj(source,target)
    native=out/'native'/f'{phantom}_{int(wavelength)}.h5'
    normalize_ipasc(out/stems[0],native,f'OpenQPAIData_InVision_{phantom}_{int(wavelength)}',str(phantom),'experimental')
    return str(native),str(out/stems[1]),json.dumps({'normalized_acquisition':str(native),
                'reference_npy':str(out/stems[1]),'material_label_map':str(out/stems[2])},indent=2)


def _ui_prepare_learning(zip_path,outdir,mua_unit,musp_unit,max_groups):
    import json
    from .experimental_learning import prepare_openqpai
    report=prepare_openqpai(zip_path,outdir,mua_unit,musp_unit,max_groups=int(max_groups) if max_groups else None)
    return report['training_corpus'],report['optical_profile'],json.dumps(report,indent=2)


def _ui_train_material(corpus,profile,weights,epochs,batch_size,seed,device="auto"):
    from .material_cvae import train
    import json
    report=train(corpus,profile,weights,int(epochs),int(batch_size),int(seed),device=device)
    return report['weights'],report['training_plot'],json.dumps(report,indent=2)


def _ui_generate_informed(engine,weights,profile,outdir,samples,seed,grid,steps,detectors,wavelength,
                           fov_mm,method,iterations,noise,noise_std,filter_on,temperature,deform,material_fractions,reconstruction_weights="",material_solver_padding_factor=1):
    import json
    from .pipeline import run_batch
    from .io import plot_preview
    # This module is deliberately NOT the human vascular CVAE.
    cfg=SimConfig.model_validate({**preset('vascular').model_dump(),
        'anatomy':'material_phantom','anatomy_engine':engine,
        'generator_weights':weights,'optical_profile':profile,
        'material_temperature':float(temperature),'material_deform_px':float(deform),
        'material_composition':[float(z) for z in material_fractions.split(',')] if str(material_fractions).strip() else None,
        'samples':int(samples),'seed':int(seed),'grid_size':int(grid),
        'material_solver_padding_factor':int(material_solver_padding_factor),
        'time_steps':int(steps),'transducers':int(detectors),
        'wavelength_nm':int(wavelength),'fov_m':float(fov_mm)/1000,
        'reconstruction_method':method,'reconstruction_iterations':int(iterations),
        'reconstruction_weights':str(reconstruction_weights or '').strip() or None,
        'detector_filter':bool(filter_on),
        'noise':{'enabled':bool(noise),'engines':['gaussian'],'relative_std':float(noise_std)}})
    folder=Path(outdir);paths=run_batch(cfg,folder)
    preview=plot_preview(paths[0]);archive=_bundle(folder,only_samples=paths)
    return str(preview),archive,json.dumps({'samples':len(paths),'raw_signal_file':str(paths[0]/'raw_signals.h5'),
        'reconstruction_h5':str(paths[0]/'reconstruction.h5'),
        'notice':'All generated acquisitions are SYNTHETIC experiment-informed data, not new physical measurements'},indent=2)


def _ui_benchmark(kind,anatomy,seed_count,grid,sensors,steps,iterations,sound,output):
    import json
    if kind=='model_mismatch':
        from .research_v05 import mismatch_benchmark
        r=mismatch_benchmark(output,anatomy=anatomy,seeds=tuple(17+12*i for i in range(int(seed_count))),
            grid=int(grid),time_steps=int(steps),transducers=int(sensors),
            iterations=int(iterations),assumed_sound_m_s=float(sound))
        picture=Path(output)/'figure_model_mismatch.png'
        result=r['summary']
    else:
        from .research import run_benchmark
        folder,r=run_benchmark(output,anatomy=anatomy,seeds=tuple(7+6*i for i in range(int(seed_count))),
            grid=int(grid),time_steps=int(steps),transducers=int(sensors),iterations=int(iterations))
        picture=Path(output)/'reconstruction_comparison.png'
        if not picture.exists():picture=Path(output)/'metric_summary.png'
        result=r['settings']
    return str(picture),_bundle(Path(output)),json.dumps({'kind':kind,'results':result,
        'caveat':'SYNTHETIC benchmark. Matched FDTD inverse is inverse crime; no empirical performance claim'},indent=2)


def _ui_train_existing_vascular(dataset,weights,epochs,batch,seed):
    from .vascular_cvae import train
    import json
    history=train(dataset,weights,epochs=int(epochs),batch_size=int(batch),seed=int(seed))
    return str(weights),json.dumps({'history':history,'data_claim':'User-curated 0/1/2/3 vessel types; not OpenQPAIData material labels'},indent=2)



def _ui_plan_datasets(task,selected_sources,profile,folder,budgets=""):
    import json
    from .dataset_wizard import plan_datasets
    caps={}
    for field in str(budgets or '').replace('\n',',').split(','):
        if not field.strip():continue
        if '=' not in field:raise ValueError('Use dataset-id=GB, dataset-id=GB for per-source limits')
        k,v=field.split('=',1)
        if k.strip() in caps:raise ValueError('Duplicate source budget')
        caps[k.strip()]=float(v.strip())
    plan=plan_datasets(task,selected_sources or [],profile,folder,source_budgets_gb=caps)
    rows=[[a['source_id'],a['filename'],a['size_gb'],('MANDATORY' if a['mandatory'] else 'optional'),
           ('already present' if a['already_present_size_checked'] else 'download') ] for a in plan['assets']]
    totals=plan['totals']
    status=(f"Profile: {plan['profile_resolved']} | Files: {totals['asset_count']} | "
            f"Remaining: {totals['remaining_download_gb']:.2f} GB | Free: {totals['free_disk_gb']:.2f} GB | "
            f"Preflight: {'PASS' if totals['preflight_ok'] else 'INSUFFICIENT SPACE'}")
    return json.dumps(plan,indent=2),rows,status


def _ui_download_dataset_plan(plan_json,accept):
    import json
    from .dataset_wizard import execute_dataset_plan
    plan=json.loads(plan_json)
    progress=None
    try:
        import gradio as gr
        progress=gr.Progress()
    except Exception:pass
    result=execute_dataset_plan(plan,bool(accept),progress=progress)
    return json.dumps(result,indent=2)



def _ui_inspect_downloaded_archives(root):
    import json
    from .dataset_wizard import inspect_downloaded_archives
    report=inspect_downloaded_archives(root,Path(root)/'archive_inventory.json')
    return json.dumps(report,indent=2)


def _ui_dis_visual(dis_zip,outdir):
    import json
    from .visualization import preview_dis_archive
    files,report=preview_dis_archive(dis_zip,outdir)
    return files,json.dumps(report,indent=2)


def _ui_svot_visual(svot_zip,phantom,slice_index,outdir):
    import json
    from .visualization import preview_svot_slice
    files,report=preview_svot_slice(svot_zip,phantom,int(slice_index),outdir)
    return files,json.dumps(report,indent=2)


def _ui_dependency_status():
    import json
    from .dependency_manager import dependency_status
    return json.dumps(dependency_status(),indent=2)


def _ui_install_dependencies(component):
    import json
    from .dependency_manager import install_selected,dependency_status
    report=install_selected(component)
    report['status_after']=dependency_status()
    return json.dumps(report,indent=2)


def _ui_prepare_multidevice(invision,tropus,svot,dis,mapping,outdir,mua_unit,musp_unit,svot_slices):
    import json
    from .multidevice_learning import prepare_openqpai_complete
    report=prepare_openqpai_complete(invision,tropus or None,svot or None,dis or None,mapping or None,
        outdir,mua_unit,musp_unit,seed=42,svot_slices_per_volume=int(svot_slices))
    return report['training_corpus'],report['optical_profile'],json.dumps(report,indent=2)


def _ui_dataset_visual_preview(archive,device,phantom,wavelength,outdir):
    import json
    from .visualization import preview_invision_sample
    files,report=preview_invision_sample(archive,phantom,int(wavelength),outdir,device=device)
    return files,json.dumps(report,indent=2)



def _ui_run_complete_workflow(plan_json,accept,run_root,mua_unit,musp_unit,svot_slices,epochs,batch_size,seed,
                              samples,grid,steps,detectors,wavelength,fov_mm,method,iterations,noise,noise_std,training_device="auto",reconstruction_weights="",material_solver_padding_factor=1):
    """One-button: download selected plan -> prepare full corpus -> train -> generate -> bundle."""
    import json
    import gradio as gr
    from .dataset_wizard import execute_dataset_plan,inspect_downloaded_archives
    from .multidevice_learning import prepare_openqpai_complete
    from .material_cvae import train
    from .pipeline import run_batch
    from .io import plot_preview
    from .visualization import preview_generated_sample
    from .config import SimConfig,preset
    if not accept:raise ValueError('Review and accept the dataset licenses in Tab 03 first.')
    if not plan_json or not str(plan_json).strip():raise ValueError('Create a dataset plan in Tab 03 first.')
    plan=json.loads(plan_json);progress=gr.Progress()
    progress(.02,desc='Checking plan and downloading selected publisher assets')
    download=execute_dataset_plan(plan,True,progress=progress)
    root=Path(plan['output_dir']);oq=root/'openqpai'
    inv=oq/'invision.zip';dis=oq/'DIS.zip';mapping=oq/'material_mapping.json'
    tropus=oq/'tropus.zip';svot=oq/'svot.zip'
    if not all(p.is_file() for p in (inv,dis,mapping)):
        raise FileNotFoundError('The selected plan must include InVision, DIS.zip and material_mapping.json for full experiment-informed generation.')
    run=Path(run_root);run.mkdir(parents=True,exist_ok=True)
    progress(.35,desc='Inspecting complete downloaded ZIP archives')
    archive_report=inspect_downloaded_archives(root,run/'archive_inventory.json')
    progress(.43,desc='Preparing multi-device material masks and optical profiles')
    prepared=prepare_openqpai_complete(inv,tropus if tropus.is_file() else None,svot if svot.is_file() else None,
        dis,mapping,run/'prepared',mua_unit,musp_unit,seed=int(seed),svot_slices_per_volume=int(svot_slices))
    weights=run/'models'/'material_cvae.pt';weights.parent.mkdir(parents=True,exist_ok=True)
    progress(.58,desc='Training material CVAE on phantom-group-disjoint data')
    training=train(prepared['training_corpus'],prepared['optical_profile'],weights,epochs=int(epochs),
                   batch_size=int(batch_size),seed=int(seed),device=training_device)
    cfg=SimConfig.model_validate({**preset('vascular').model_dump(),
        'anatomy':'material_phantom','anatomy_engine':'material_cvae','generator_weights':str(weights),
        'optical_profile':prepared['optical_profile'],'samples':int(samples),'seed':int(seed),
        'grid_size':int(grid),'time_steps':int(steps),'transducers':int(detectors),
        'material_solver_padding_factor':int(material_solver_padding_factor),
        'wavelength_nm':int(wavelength),'fov_m':float(fov_mm)/1000,
        'reconstruction_method':method,'reconstruction_iterations':int(iterations),
        'reconstruction_weights':str(reconstruction_weights or '').strip() or None,
        'detector_filter':False if method in ('iterative','adjoint','adjoint_fdtd','fista','time_reversal','iterative_time_reversal') else True,
        'noise':{'enabled':bool(noise),'engines':['gaussian'],'relative_std':float(noise_std)}})
    generated=run/'generated'
    progress(.72,desc='Generating new synthetic detector signals and reconstructed images')
    paths=run_batch(cfg,generated,callback=lambda i,n,p:progress(.72+.20*i/max(n,1),desc=f'Generated sample {i}/{n}'))
    overview=plot_preview(paths[0]);gallery=preview_generated_sample(paths[0],run/'visuals')
    # Include the complete download/ingestion/training/generation audit in the final ZIP.
    (run/'full_workflow_report.json').write_text(json.dumps({'download':download,'archive_inventory':str(run/'archive_inventory.json'),
        'preparation':prepared,'training':training,'generated_samples':len(paths),
        'generated_sample_folders':[str(p) for p in paths],
        'scientific_notice':'Generated acquisitions are synthetic and experiment-informed, not new experimental measurements. Digital Twin QA archive is kept separate from material-CVAE training.'},indent=2),encoding='utf-8')
    archive=_bundle(run)
    report={'download_status':{k:v['status'] for k,v in download['sources'].items()},
        'archive_count':archive_report['archive_count'],'preparation':prepared,'training':training,
        'generated_samples':len(paths),'output_folder':str(run),
        'notice':'Synthetic experiment-informed data only; full-reference QA and real-data calibration remain separate.'}
    progress(1.0,desc='Complete workflow finished')
    return str(overview),gallery,archive,json.dumps(report,indent=2)


def _ui_generate_informed_v07(*args):
    import json
    import os
    import shutil
    import tempfile
    from .visualization import preview_generated_sample
    overview,archive,meta=_ui_generate_informed(*args)
    info=json.loads(meta);sample=Path(args[3])/'material_phantom_00000'
    gallery=preview_generated_sample(sample)
    # Gradio rejects file outputs from arbitrary persistent Results folders.
    # Stage only the files from THIS generated run in the OS temporary folder,
    # rather than exposing the whole Results tree (which may contain held-out data).
    staging=Path(tempfile.mkdtemp(prefix='paforge_ui_media_'))
    def stage_for_gradio(path):
        source=Path(path)
        target=staging/source.name
        try:
            os.link(source,target)  # Cheap when source and temp are on one filesystem.
        except OSError:
            shutil.copy2(source,target)
        return str(target)
    return stage_for_gradio(overview),[stage_for_gradio(x) for x in gallery],stage_for_gradio(archive),meta


def _ui_catalog_details(source_id):
    import json
    from .dataset_registry import describe
    return json.dumps(describe(source_id),ensure_ascii=False,indent=2)


def _ui_portal_inspect(source_id,path):
    import json
    from .dataset_portal import inspect_local_file
    return json.dumps(inspect_local_file(source_id,path),ensure_ascii=False,indent=2)


def _ui_portal_visualize(source_id,path,key,member,index,semantic,outdir):
    import json
    from .dataset_portal import visualize_array
    visuals,report=visualize_array(source_id,path,key,outdir,int(index),member or None,semantic)
    return visuals,json.dumps(report,ensure_ascii=False,indent=2)


def _ui_skin_pairs(path,outdir,limit):
    import json
    from .dataset_portal import extract_skin_pairs
    report=extract_skin_pairs(path,outdir,int(limit))
    preview=str(Path(outdir)/'paired_preview.png')
    return preview,json.dumps(report,ensure_ascii=False,indent=2)


def _ui_two_links_download(folder,licenses):
    """Independent, truthful per-publisher download results with no guessed assets."""
    import json
    from .dataset_wizard import plan_datasets,execute_dataset_plan
    if not licenses:raise ValueError('Review and accept both publisher license terms before download')
    results={}
    for sid in ('cambridge-patato','zenodo-23348355'):
        try:
            plan=plan_datasets('custom',[sid],'complete',folder)
            if not plan['totals']['preflight_ok']:
                results[sid]={'status':'insufficient_disk','remaining_gb':plan['totals']['remaining_download_gb']}
                continue
            report=execute_dataset_plan(plan,True)
            results[sid]={'status':report['sources'].get(sid,{}).get('status'),
                          'details':report['sources'].get(sid,{}),
                          'publisher_report_path':report.get('report_path')}
        except Exception as exc:
            results[sid]={'status':'publisher_discovery_or_download_error',
                          'error':str(exc),
                          'note':'Source not certified until publisher metadata and checksums are available.'}
    return json.dumps(results,indent=2)


def _ui_patato_normalize(source,out,frame,wave_idx,group,confirm):
    import json
    from .patato_bridge import normalize_patato
    rep=normalize_patato(source,out,int(frame),int(wave_idx),'experimental' if confirm else 'unknown',
                         bool(confirm),str(group))
    return rep['normalized_acquisition'],json.dumps(rep,indent=2)


def _ui_patato_reference(source,dest,frame,wave_idx,key):
    import json
    from .patato_bridge import extract_patato_reference
    report=extract_patato_reference(source,dest,int(frame),int(wave_idx),key)
    return report['reference_npy'],json.dumps(report,indent=2)


def _ui_generic_import(sid,path,key,out,det_axis,time_axis,fixed,fs,wavelength,positions,group,confirm):
    import json
    from .dataset_portal import normalize_source_signals
    idx={}
    for field in str(fixed or '').split(','):
        if not field.strip():continue
        ax,i=field.split('=',1)
        idx[int(ax.strip())]=int(i.strip())
    report=normalize_source_signals(sid,path,key,out,int(det_axis),int(time_axis),idx,
                    float(fs),float(wavelength),positions,group,bool(confirm))
    return report['native_h5'],json.dumps(report,indent=2)


def _ui_compare_shared(source,uploaded,method_names,grid,fov_mm,sound,iterations,reg,decimation,weights,truth):
    import json,tempfile
    from .reconstruction_lab import run_comparison
    src=str(source or '').strip().strip('"') or str(uploaded or '')
    if not src or not Path(src).is_file():raise ValueError('Select an existing synthetic or experimental normalized HDF5 file')
    folder=Path(tempfile.mkdtemp(prefix='paforge_comparison09_'))
    report=run_comparison(src,folder,method_names,int(grid),float(fov_mm)/1000,float(sound),
            int(iterations),float(reg),int(decimation),str(weights or '').strip() or None,
            str(truth or '').strip() or None)
    gallery=[str(folder/'input_signals.png')]
    if (folder/'comparison.png').exists():gallery.append(str(folder/'comparison.png'))
    gallery.extend(str(folder/(r['method']+'.png')) for r in report['methods'] if r['status']=='ok')
    return gallery,json.dumps(report,indent=2),_bundle(folder)


def _ui_train_reconstructor(paired,weights,architecture,epochs,batch,seed,device):
    import json
    from .learned_reconstruction import train_paired
    report=train_paired(paired,weights,architecture,int(epochs),int(batch),int(seed),device)
    return report['weights'],json.dumps(report,indent=2)


def build_ui():
    import gradio as gr
    with gr.Blocks(title="PA-FORGE v0.9 | Unified Reconstruction Research Workbench") as demo:
        with gr.Tab("00 · Setup & Dependencies"):
            gr.Markdown("# PA-FORGE v0.8 · Setup Center\nCheck the local Python environment and install optional components with one click. The initial dashboard bootstrap is provided by `START_PA_FORGE.bat`; this tab installs optional training dependencies after the UI is open.")
            dep_refresh=gr.Button("Check installed dependencies")
            dep_status=gr.Code(language="json",label="Python package status",value="Click Check installed dependencies")
            dep_component=gr.Dropdown([("Dashboard UI (Gradio)","ui"),("Machine learning · PyTorch CPU", "training_cpu"),("UI + machine learning · CPU", "all_cpu"),("Official PATATO Python backend", "patato")],value="training_cpu",label="Component to install")
            dep_install=gr.Button("Install / repair selected component",variant="primary")
            dep_log=gr.Code(language="json",label="Installation log")
            dep_refresh.click(_ui_dependency_status,outputs=dep_status)
            dep_install.click(_ui_install_dependencies,inputs=dep_component,outputs=dep_log)
            gr.Markdown("**GPU training:** select CUDA/Auto in Tab 04 if a compatible CUDA-enabled PyTorch is already installed. The installer intentionally does not install/overwrite a GPU build blindly; see the official PyTorch CUDA selector. A selected CUDA device never silently falls back to CPU.")
            gr.Markdown("**Note:** PyTorch CPU installation can be large. Installation requires internet access and may take several minutes. Restart PA-FORGE after installation. CUDA-specific PyTorch setup is intentionally not guessed; use the official PyTorch selector for hardware-specific builds.")
        with gr.Tab("01 · Synthetic Generator"):
            gr.Markdown("# PA-FORGE · Photoacoustic Data Generator\n**Local research prototype.** 2D approximate physics; no experimental realism certification.")
            with gr.Row():
                with gr.Column(scale=2):
                    anatomy=gr.Dropdown(["forearm","skin","breast","carotid","vascular"],value="vascular",label="Anatomical region")
                    samples=gr.Slider(1,25,value=1,step=1,label="Samples in this interactive run")
                    seed=gr.Number(value=42,precision=0,label="Reproducibility seed")
                    with gr.Accordion("Vascular network (arteries, veins, capillaries)",open=True):
                        vascular_pattern=gr.Dropdown(["branching","paired","plexus"],value="branching",label="Network pattern (vascular mode)")
                        vascular_branches=gr.Slider(1,5,value=3,step=1,label="Branch levels")
                        vascular_density=gr.Slider(.2,3,value=1,step=.1,label="Capillary density multiplier")
                    with gr.Accordion("Reconstruction",open=True):
                        recon_method=gr.Dropdown(list(RECON_METHODS),value="ubp",label="Reconstruction method")
                        recon_iterations=gr.Slider(1,50,value=8,step=1,label="Iterative solver steps")
                        recon_weights=gr.Textbox(value="",label="Optional trained reconstruction checkpoint (.pt) for learned methods")
                        filter_enabled=gr.Checkbox(value=True,label="Apply detector bandpass (turn OFF for matched iterative inversion)")
                    with gr.Accordion("Learned vs procedural anatomy",open=False):
                        anatomy_engine=gr.Dropdown(["procedural","cvae","vascular_cvae"],value="procedural",label="Anatomy generator")
                        generator_weights=gr.Textbox(value="",label="Trusted local model weights path (required for cVAE)")
                    with gr.Accordion("Noise engines · completely optional",open=True):
                        noise=gr.Checkbox(value=False,label="Enable added noise (OFF by default)")
                        engines=gr.CheckboxGroup(["gaussian","colored","impulsive","drift","empirical"],value=["gaussian"],label="Active engines (only run when enabled)")
                        strength=gr.Slider(0,0.3,value=.025,step=.005,label="Noise SD relative to clean waveform RMS")
                        empirical=gr.Textbox(label="Trusted local empirical noise path (required when engine selected)",value="")
                    with gr.Accordion("Acquisition and device settings",open=False):
                        illumination=gr.Dropdown(["attenuation_proxy","uniform_idealized"],value="uniform_idealized",label="Optical fluence model (uniform is idealized; scattering ignored)")
                        wavelength=gr.Slider(660,1300,value=800,step=10,label="Laser wavelength (nm)")
                        transducers=gr.Slider(8,128,value=64,step=8,label="Transducer count")
                        geometry=gr.Dropdown(["circle","arc","linear"],value="circle",label="Acquisition geometry")
                        center=gr.Slider(.4,3,value=1.8,step=.1,label="Detector center frequency (MHz)")
                        bandwidth=gr.Slider(.3,3,value=1.8,step=.1,label="Detector bandwidth (MHz)")
                        sound=gr.Slider(0,2,value=1,step=.1,label="Sound speed anatomical variation multiplier")
                        scattering=gr.Slider(.1,3,value=1,step=.1,label="Optical reduced scattering multiplier")
                        laser=gr.Slider(0,.2,value=0,step=.005,label="Pulse energy jitter fraction")
                        drop=gr.Slider(0,.5,value=0,step=.01,label="Channel dropout probability")
                        grid=gr.Dropdown([64,96,128],value=96,label="Spatial grid points per dimension")
                        steps=gr.Slider(200,800,value=380,step=20,label="Time steps")
                    # Keep anatomy-specific geometry defaults when switching regions.
                    anatomy.change(fn=lambda a: gr.update(value=preset(a).geometry),inputs=anatomy,outputs=geometry)
                    anatomy.change(fn=lambda a: gr.update(value=preset(a).illumination_model),inputs=anatomy,outputs=illumination)
                    recon_method.change(fn=lambda m: gr.update(value=False) if m in ('iterative','adjoint','time_reversal','adjoint_fdtd','fista','iterative_time_reversal') else gr.update(),
                                        inputs=recon_method,outputs=filter_enabled)
                    # Only a well-documented default geometry should change with anatomy.
                    button=gr.Button("Generate simulated signals and images",variant="primary")
                with gr.Column(scale=3):
                    gr.Markdown("### Outputs · raw signals and images stored separately")
                    preview=gr.Image(type="filepath",label="Initial pressure · measured traces · image reconstruction")
                    archive=gr.File(label="ZIP containing HDF5 raw signals, reconstructed image, and truth")
                    metadata=gr.Code(label="Reproducible configuration",language="json")
            button.click(fn=generate_ui,inputs=[anatomy,samples,seed,noise,engines,strength,empirical,
                           wavelength,transducers,geometry,center,bandwidth,sound,scattering,laser,drop,grid,steps,
                           vascular_pattern,vascular_branches,vascular_density,recon_method,recon_iterations,filter_enabled,
                           illumination,anatomy_engine,generator_weights,recon_weights],
                         outputs=[preview,archive,metadata])
            gr.Markdown("**Important:** No added noise ≠ noiseless real hardware. IPASC conformance and patient-data realism remain unvalidated.")
        with gr.Tab("02 · Experimental Reconstruction"):
            gr.Markdown("### Reconstruct measured acquisitions | 2D · qualitative\nUse **normalized PA-FORGE HDF5** with measured positions. For original IPASC files, run the existing `extract-acquisition` command first. Orientation controls affect **display only**, not raw measurements or reconstructed numeric data.")
            with gr.Row():
                with gr.Column(scale=2):
                    real_path=gr.Textbox(label="Normalized acquisition HDF5 (local path)",placeholder="data/real/invision_P.5.10.2_700.h5")
                    real_upload=gr.File(label="Or upload normalized HDF5",file_types=[".h5",".hdf5"],type="filepath")
                    reference_path=gr.Textbox(label="Optional reference reconstruction (.npy, local path)",placeholder="data/selected_invision/invision/recon/P.5.10.2_700.npy")
                    with gr.Row():
                        real_method=gr.Dropdown(list(RECON_METHODS),value="ubp",label="Method")
                        real_grid=gr.Dropdown([64,96,128,192,256],value=256,label="Reconstruction grid")
                    with gr.Row():
                        real_fov=gr.Slider(5,150,value=60,step=1,label="FOV (mm); provisional until calibrated")
                        real_speed=gr.Slider(1000,2200,value=1500,step=10,label="Assumed sound speed (m/s)")
                    with gr.Accordion("Iterative FISTA / experimental advanced controls",open=False):
                        gr.Markdown("For **FISTA/adjoint FDTD**, enlarge FOV to include **all detectors** (InVision uses an ~81-mm-wide arc; use ≥100 mm) and consider 64–96 grid points initially. This is a slow approximate 2D homogeneous model, not calibrated device physics.")
                        real_iterations=gr.Slider(1,40,value=10,step=1,label="FISTA iterations")
                        real_reg=gr.Slider(0,.1,value=.001,step=.001,label="L2 regularization (relative)")
                        real_weights=gr.Textbox(value="",label="Trained .pt checkpoint for learned methods")
                        real_decimate=gr.Dropdown([1,2,4,8,16],value=4,label="Anti-aliased temporal decimation (FDTD only)")
                    gr.Markdown("**Image orientation** (preview-only; never changes the HDF5 signal or reconstructed numerical image)")
                    with gr.Row():
                        rotation=gr.Dropdown([0,90,180,270],value=0,label="Counterclockwise rotation (°)")
                        flip_lr=gr.Checkbox(value=False,label="Flip left ↔ right")
                        flip_ud=gr.Checkbox(value=False,label="Flip up ↕ down")
                    inspect=gr.Button("Inspect acquisition metadata")
                    real_run=gr.Button("Reconstruct acquisition",variant="primary")
                    real_meta=gr.Code(language="json",label="Run / acquisition metadata")
                with gr.Column(scale=3):
                    with gr.Row():
                        real_image=gr.Image(label="PA-FORGE reconstruction (display orientation)",type="numpy")
                        real_reference=gr.Image(label="Published reference (unchanged orientation)",type="numpy")
                    gr.Markdown("Reference images can have a different field of view and pixel scale. Side-by-side does **not** constitute registered or quantitative comparison.")
                    real_zip=gr.File(label="Download reconstructed numeric HDF5 + original PNG + manifest + settings")
            rec_state=gr.State(None)
            ref_state=gr.State(None)
            inspect.click(fn=_summarize_acquisition,inputs=[real_path,real_upload],outputs=real_meta)
            real_run.click(fn=_experimental_dashboard_reconstruct,inputs=[real_path,real_upload,reference_path,real_method,real_fov,real_grid,real_speed,rotation,flip_lr,flip_ud,real_iterations,real_reg,real_decimate,real_weights],outputs=[real_image,real_reference,real_meta,real_zip,rec_state,ref_state])
            for transform in (rotation,flip_lr,flip_ud):
                transform.change(fn=_display_orient,inputs=[rec_state,ref_state,rotation,flip_lr,flip_ud],outputs=[real_image,real_reference])
        with gr.Tab("03 · Intelligent Dataset Hub"):
            gr.Markdown("## Choose your research objective and datasets\nPA-FORGE discovers publisher assets, enforces mandatory sources for the selected task, estimates download size, checks free disk space and creates a resumable download plan. Complete archives are retained as ZIP files; the software does not blindly unpack 20 GB archives.")
            from .dataset_hub import SOURCES
            catalog=[(x.title+" ("+x.id+")",x.id) for x in SOURCES]
            with gr.Row():
                with gr.Column(scale=2):
                    ds_task=gr.Dropdown([("Experiment-informed material/optical generation","experiment_informed_generation"),("Complete research corpus · OpenQPAIData + Digital Twin","full_research_corpus"),("Digital Twin full-reference QA only","digital_twin_qa"),("Custom dataset selection","custom")],value="full_research_corpus",label="What do you want to do?")
                    ds_sources=gr.Dropdown(catalog,value=["openqpai","digital-twin-iqa"],multiselect=True,label="Datasets you want to include")
                    ds_profile=gr.Dropdown([("Auto · choose largest profile that fits available disk","auto"),("Starter · smallest useful sample","starter"),("Recommended · all 2D optical/raw systems, excludes 19.8 GB SVOT","recommended"),("Complete · every public asset in selected datasets","complete")],value="complete",label="How much data should PA-FORGE acquire?")
                    ds_folder=gr.Textbox(value="data/external",label="Download destination")
                    ds_caps=gr.Textbox(value="",label="Optional per-dataset caps (GB, whole files only)",placeholder="paiskintone=3.5, synthetic-skin=0.5")
                    ds_plan_button=gr.Button("1 · Discover datasets and calculate download plan",variant="primary")
                    ds_plan_status=gr.Textbox(label="Storage preflight",interactive=False)
                    ds_accept=gr.Checkbox(value=False,label="I reviewed the publisher licenses and accept their terms")
                    ds_download_button=gr.Button("2 · Download planned datasets",variant="primary")
                with gr.Column(scale=3):
                    ds_table=gr.Dataframe(headers=["Source","File","GB","Requirement","Status"],label="Selected publisher assets",interactive=False)
                    ds_plan_json=gr.Code(language="json",label="Full download plan (also used by Download)")
                    ds_log=gr.Code(language="json",label="Download progress and final report")
            ds_plan_button.click(_ui_plan_datasets,[ds_task,ds_sources,ds_profile,ds_folder,ds_caps],[ds_plan_json,ds_table,ds_plan_status])
            ds_download_button.click(_ui_download_dataset_plan,[ds_plan_json,ds_accept],[ds_log])
            with gr.Accordion("Inspect complete downloaded archives without extracting them",open=False):
                archive_root=gr.Textbox(value="data/external",label="Downloaded dataset root")
                archive_inspect_button=gr.Button("Inspect ZIP contents and file roles")
                archive_inspect_report=gr.Code(language="json",label="Archive inventory")
                archive_inspect_button.click(_ui_inspect_downloaded_archives,archive_root,archive_inspect_report)
            gr.Markdown("**Mandatory optical/material inputs:** for experiment-informed generation, OpenQPAIData is automatically included. The full optical/material workflow requires `DIS.zip`, `invision.zip`, and `material_mapping.json`; TROPUS adds a second raw-measurement system. The complete OpenQPAIData profile also downloads `svot.zip` (19.8 GB), which contains 3D reconstructions/labels/optical volumes but no shared raw SVOT time series. The full research-corpus option additionally downloads the Digital Twin QA archive. Approximate combined complete download: 23.4 GB, before any extraction.")
            gr.Markdown("### Select one OpenQPAIData InVision phantom without unpacking entire ZIP")
            with gr.Row():
                raw_zip=gr.Textbox(value="data/external/openqpai/invision.zip",label="InVision ZIP local path")
                phantom=gr.Textbox(value="P.5.10.2",label="Phantom ID")
                wavelength=gr.Dropdown([700,750,800,850,900],value=700,label="Wavelength (nm)")
                select_dir=gr.Textbox(value="data/selected_invision",label="Extract/output folder")
            confirm_origin=gr.Checkbox(value=False,label="Confirm publisher identifies InVision files as measured phantom acquisitions")
            select_button=gr.Button("Extract selected phantom + normalize IPASC",variant="primary")
            with gr.Row():
                selected_acq=gr.Textbox(label="Normalized HDF5 path (copy to Experimental Reconstruction)")
                selected_ref=gr.Textbox(label="Publisher reference image path")
            select_log=gr.Code(label="Extraction report",language="json")
            select_button.click(_ui_select_invision,[raw_zip,phantom,wavelength,select_dir,confirm_origin],
                                [selected_acq,selected_ref,select_log]).then(
                                    lambda acq,ref:(acq,ref),[selected_acq,selected_ref],[real_path,reference_path])
            gr.Markdown("### Import any planar IPASC HDF5 using its actual detector geometry and sampling metadata")
            with gr.Row():
                raw_input=gr.Textbox(label="Raw IPASC HDF5 local path")
                raw_upload=gr.File(label="Or upload raw IPASC",file_types=[".h5",".hdf5"],type="filepath")
                native_output=gr.Textbox(value="data/real/imported_acquisition.h5",label="Normalized output HDF5")
            with gr.Row():
                source_id=gr.Textbox(value="user_supplied_IPASC",label="Source ID")
                group_id=gr.Textbox(value="phantom_001",label="Independent phantom/subject ID")
                origin=gr.Dropdown(["experimental","synthetic","unknown"],value="unknown",label="Source provenance")
            confirm_prov=gr.Checkbox(value=False,label="I attest source documentation confirms an experimental acquisition")
            with gr.Row():
                import_button=gr.Button("Normalize IPASC (metadata-driven)")
                inventory_button=gr.Button("Inspect raw HDF5")
            import_path=gr.Textbox(label="Normalized output path")
            import_log=gr.Textbox(label="Importer message")
            inventory_log=gr.Code(language="json",label="HDF5 inventory (truncated)")
            import_button.click(_ui_import_ipasc,[raw_input,raw_upload,native_output,source_id,group_id,origin,confirm_prov],
                                [import_path,import_log])
            inventory_button.click(_ui_inspect_source,[raw_input,raw_upload],[inventory_log])
        with gr.Tab("04 · Learn from OpenQPAIData"):
            gr.Markdown("## 04 — Learn phantom-material and optical distributions\nThe complete workflow can use InVision + TROPUS 2D masks and selected 2D slices from SVOT 3D label volumes. Train/validation/test separation is by phantom ID across devices. InVision/TROPUS mapped `mua`/`musp` arrays provide wavelength-specific optical profiles. DIS is independently inventoried as optical reference data; its NPZ semantics and material mapping are preserved for audit rather than silently guessed.")
            with gr.Accordion("Complete multi-device corpus preparation (recommended for full dataset)",open=True):
                with gr.Row():
                    full_invision=gr.Textbox(value="data/external/openqpai/invision.zip",label="InVision ZIP (required)")
                    full_tropus=gr.Textbox(value="data/external/openqpai/tropus.zip",label="TROPUS ZIP (optional second device)")
                with gr.Row():
                    full_svot=gr.Textbox(value="data/external/openqpai/svot.zip",label="SVOT ZIP (optional 3D labels; 19.8 GB)")
                    full_dis=gr.Textbox(value="data/external/openqpai/DIS.zip",label="DIS ZIP (required for complete optical audit)")
                    full_mapping=gr.Textbox(value="data/external/openqpai/material_mapping.json",label="Material mapping JSON")
                with gr.Row():
                    full_out=gr.Textbox(value="models/openqpai_complete",label="Corpus output folder")
                    full_mua=gr.Dropdown(["1/cm","1/m"],value="1/cm",label="Mapped mua source unit")
                    full_musp=gr.Dropdown(["1/cm","1/m"],value="1/cm",label="Mapped musp source unit")
                    full_slices=gr.Slider(2,12,value=6,step=1,label="SVOT slices sampled per volume")
                full_prepare=gr.Button("Prepare complete multi-device material + optical corpus",variant="primary")
            with gr.Row():
                with gr.Column():
                    learn_zip=gr.Textbox(value="data/external/openqpai/invision.zip",label="Downloaded InVision ZIP")
                    learn_folder=gr.Textbox(value="models/openqpai",label="Prepared corpus/output folder")
                    mua_unit=gr.Dropdown(["1/m","1/cm"],value="1/cm",label="mua source units (OpenQPAIData: 1/cm, publisher verified)")
                    musp_unit=gr.Dropdown(["1/m","1/cm"],value="1/cm",label="musp source units (OpenQPAIData: 1/cm, publisher verified)")
                    learn_limit=gr.Number(value=0,precision=0,label="Max phantom groups (0 = all)")
                    learn_prepare=gr.Button("Prepare train-only optical + material corpus",variant="primary")
                with gr.Column():
                    corpus_path=gr.Textbox(label="Training material corpus NPZ")
                    optical_path=gr.Textbox(label="Training-only optical profile JSON")
                    corpus_report=gr.Code(label="Corpus and splits",language="json")
            learn_prepare_event=learn_prepare.click(_ui_prepare_learning,[learn_zip,learn_folder,mua_unit,musp_unit,learn_limit],
                                [corpus_path,optical_path,corpus_report])
            full_prepare.click(_ui_prepare_multidevice,[full_invision,full_tropus,full_svot,full_dis,full_mapping,full_out,full_mua,full_musp,full_slices],
                               [corpus_path,optical_path,corpus_report])
            with gr.Accordion("One-click complete workflow · download → prepare → train → generate",open=False):
                gr.Markdown("Create and review the complete plan in Tab 03 first. This action will verify/download its assets, inventory all ZIPs, prepare the multi-device corpus, train the material CVAE and generate a small synthetic batch. It can take a long time for the complete 23.4 GB corpus.")
                full_run_root=gr.Textbox(value="outputs/full_dataset_run",label="Complete run output folder")
                with gr.Row():
                    full_epochs=gr.Slider(1,150,value=30,step=1,label="CVAE epochs")
                    full_batch=gr.Dropdown([2,4,8,16],value=8,label="Batch size")
                    full_samples=gr.Slider(1,100,value=10,step=1,label="Generated samples")
                    full_seed=gr.Number(value=42,precision=0,label="Seed")
                    full_device=gr.Dropdown(["cpu","auto","cuda","mps"],value="cpu",label="CVAE training device (CPU for portable runs; GPU opt-in)")
                with gr.Row():
                    full_grid=gr.Dropdown([64,96,128],value=64,label="Simulation grid")
                    full_steps=gr.Slider(100,900,value=380,step=10,label="Acoustic time steps")
                    full_detectors=gr.Slider(8,256,value=48,step=8,label="Synthetic detector count")
                    full_wave=gr.Dropdown([700,750,800,850,900],value=700,label="Wavelength (nm)")
                with gr.Row():
                    full_fov=gr.Slider(10,150,value=70,step=1,label="FOV (mm)")
                    full_method=gr.Dropdown(list(RECON_METHODS),value="ubp",label="Reconstruction method")
                    full_iterations=gr.Slider(1,40,value=8,step=1,label="FISTA iterations")
                    full_noise=gr.Checkbox(value=False,label="Add Gaussian noise")
                    full_noise_std=gr.Slider(0,.2,value=.025,step=.005,label="Noise relative SD")
                    full_recon_weights=gr.Textbox(value="",label="Optional learned reconstruction checkpoint")
                    full_padding=gr.Dropdown([2,1],value=2,label="Experimental material propagation-domain padding (2 = clearance check; 1 = legacy)")
                full_run_button=gr.Button("Run complete research workflow",variant="primary")
                full_run_image=gr.Image(type="filepath",label="First generated sample overview")
                full_run_gallery=gr.Gallery(label="Generated image and signal diagnostics",columns=3,height=540)
                full_run_zip=gr.File(label="Complete workflow ZIP · download, preparation, model, samples and charts")
                full_run_report=gr.Code(language="json",label="Complete workflow report")
                full_run_button.click(_ui_run_complete_workflow,[ds_plan_json,ds_accept,full_run_root,full_mua,full_musp,full_slices,
                    full_epochs,full_batch,full_seed,full_samples,full_grid,full_steps,full_detectors,full_wave,full_fov,
                    full_method,full_iterations,full_noise,full_noise_std,full_device,full_recon_weights,full_padding],[full_run_image,full_run_gallery,full_run_zip,full_run_report])
            gr.Markdown("### Train a categorical material CVAE (independent of human vascular CVAE)")
            with gr.Row():
                with gr.Column():
                    training_weights=gr.Textbox(value="models/openqpai/material_cvae.pt",label="Trusted local model checkpoint (.pt)")
                    training_epochs=gr.Slider(1,150,value=20,step=1,label="Training epochs")
                    training_batch=gr.Dropdown([1,2,4,8,16],value=8,label="Batch size")
                    training_seed=gr.Number(value=42,precision=0,label="Training seed")
                    training_device=gr.Dropdown(["cpu","auto","cuda","mps"],value="cpu",label="Training device (CPU for reproducibility; GPU opt-in)")
                    train_button=gr.Button("Train CVAE with phantom-disjoint validation",variant="primary")
                with gr.Column():
                    trained_model=gr.Textbox(label="Trained model path")
                    training_curve=gr.Image(type="filepath",label="Training/held-out validation loss")
                    training_report=gr.Code(label="Loss history, source and group split",language="json")
            train_event=train_button.click(_ui_train_material,[corpus_path,optical_path,training_weights,training_epochs,training_batch,training_seed,training_device],
                               [trained_model,training_curve,training_report])
        with gr.Tab("05 · Experiment-informed Generation"):
            gr.Markdown("## 05 — NEW experiment-informed synthetic datasets\nUse trained material CVAE **or** train-only empirical material-mask resampling. Each sample includes **ground truth + raw detector signals (HDF5) + reconstructed image (HDF5/PNG) + per-sample manifest**. This does **not** create experimental measurements.")
            with gr.Row():
                with gr.Column(scale=2):
                    gen_engine=gr.Dropdown(["empirical_material","material_cvae"],value="empirical_material",label="Generator")
                    gen_source=gr.Textbox(value="",label="Required: absolute path to material_corpus.npz (or CVAE .pt)")
                    gen_profile=gr.Textbox(value="",label="Required: absolute path to train-only optical_profile.json")
                    gen_output=gr.Textbox(value="outputs/informed",label="Destination folder (persistent)")
                    with gr.Row():
                        gen_n=gr.Slider(1,200,value=2,step=1,label="New samples")
                        gen_seed=gr.Number(value=42,precision=0,label="Seed")
                        gen_wave=gr.Dropdown([700,750,800,850,900],value=700,label="Observed wavelength (nm)")
                    with gr.Row():
                        gen_grid=gr.Dropdown([64,96,128,192],value=64,label="2D grid")
                        gen_steps=gr.Slider(100,900,value=380,step=10,label="Simulation steps")
                        gen_sensors=gr.Slider(8,256,value=48,step=8,label="Simulated detector count")
                        gen_fov=gr.Slider(10,150,value=70,step=1,label="FOV (mm)")
                        gen_padding=gr.Dropdown([2,1],value=2,label="Material propagation-domain padding (2 = exploratory clearance; 1 = legacy geometry)")
                    with gr.Row():
                        gen_method=gr.Dropdown(list(RECON_METHODS),value="ubp",label="Reconstruction")
                        gen_iters=gr.Slider(1,40,value=8,step=1,label="FISTA iterations")
                        gen_filter=gr.Checkbox(value=False,label="Detector filter (OFF for FDTD inversion)")
                        gen_recon_weights=gr.Textbox(value="",label="Trained reconstruction checkpoint for learned methods")
                    with gr.Row():
                        gen_noise=gr.Checkbox(value=False,label="Add noise (OFF default)")
                        gen_noise_std=gr.Slider(0,.2,value=.025,step=.005,label="Optional Gaussian relative SD")
                        gen_temp=gr.Slider(.2,2,value=.9,step=.1,label="CVAE temperature")
                        gen_deform=gr.Slider(0,5,value=1,step=.25,label="Empirical deformation (pixels)")
                    gen_fractions=gr.Textbox(value="",label="Optional material area fractions (e.g. 0.2,0.6,0.1,0.1). Empty=learned training composition")
                    gen_button=gr.Button("Generate NEW synthetic signals + images",variant="primary")
                with gr.Column(scale=3):
                    gen_preview=gr.Image(label="Ground-truth pressure • raw traces • reconstruction overview",type="filepath")
                    gen_gallery=gr.Gallery(label="Generated data diagnostics · initial pressure · labels · reconstructed image · raw sinogram · time traces · distributions",columns=3,height=620)
                    gen_zip=gr.File(label="Download complete dataset ZIP")
                    gen_log=gr.Code(language="json",label="Output manifest")
            gr.Markdown("**Research guardrail:** padding=2 checks receiver clearance for synthetic material phantoms; it is not evidence of experimental RF fidelity. Keep the independent test groups sealed.")
            gen_button.click(_ui_generate_informed_v07,[gen_engine,gen_source,gen_profile,gen_output,gen_n,gen_seed,
                gen_grid,gen_steps,gen_sensors,gen_wave,gen_fov,gen_method,gen_iters,gen_noise,gen_noise_std,gen_filter,
                gen_temp,gen_deform,gen_fractions,gen_recon_weights,gen_padding],[gen_preview,gen_gallery,gen_zip,gen_log])

        with gr.Tab("06 · Benchmarks & Vascular CVAE"):
            gr.Markdown("## 06 — Reproducible synthetic reconstruction benchmarking\nBoth benchmarks share the numerical forward model. The matched iterative baseline is an **inverse crime** and cannot establish experimental accuracy.")
            with gr.Row():
                with gr.Column(scale=2):
                    bench_kind=gr.Dropdown(["model_mismatch","matched_synthetic"],value="model_mismatch",label="Benchmark")
                    bench_anatomy=gr.Dropdown(["vascular","forearm","skin","breast","carotid"],value="vascular",label="Synthetic anatomy")
                    bench_seed=gr.Slider(1,10,value=2,step=1,label="Distinct seeds")
                    bench_grid=gr.Dropdown([48,64,96],value=48,label="Grid")
                    bench_sensor=gr.Slider(8,96,value=20,step=4,label="Detector count")
                    bench_steps=gr.Slider(100,420,value=175,step=5,label="Time steps")
                    bench_iterations=gr.Slider(1,30,value=8,step=1,label="FISTA iterations")
                    bench_sound=gr.Slider(1200,1900,value=1630,step=10,label="Assumed speed for mismatch (m/s)")
                    bench_out=gr.Textbox(value="experiments/dashboard_benchmark",label="Benchmark folder")
                    bench_go=gr.Button("Run synthetic benchmark",variant="primary")
                with gr.Column(scale=3):
                    bench_img=gr.Image(type="filepath",label="Benchmark image")
                    bench_zip=gr.File(label="Metrics + figure + reproducibility report ZIP")
                    bench_report=gr.Code(language="json",label="Benchmark results")
            bench_go.click(_ui_benchmark,[bench_kind,bench_anatomy,bench_seed,bench_grid,bench_sensor,
                bench_steps,bench_iterations,bench_sound,bench_out],[bench_img,bench_zip,bench_report])
            gr.Markdown("### Optional separate vascular CVAE training\n**Only** for an independently curated vessel-type NPZ containing `types [N,64,64]` with 0=background, 1=artery, 2=vein, 3=capillary. InVision material labels are **not** valid here.")
            with gr.Row():
                curated=gr.Textbox(label="Curated vessel-type training NPZ",placeholder="data/curated_vessels.npz")
                vascular_weights=gr.Textbox(value="models/vascular_cvae.pt",label="Weights output .pt")
                vascular_epochs=gr.Slider(1,100,value=20,step=1,label="Epochs")
                vascular_batch=gr.Dropdown([2,4,8,12,16],value=8,label="Batch size")
                vascular_seed=gr.Number(value=42,precision=0,label="Seed")
            vascular_button=gr.Button("Train true vessel-class CVAE")
            vascular_final=gr.Textbox(label="Vascular checkpoint path")
            vascular_log=gr.Code(language="json",label="Training result")
            vascular_button.click(_ui_train_existing_vascular,[curated,vascular_weights,vascular_epochs,
                vascular_batch,vascular_seed],[vascular_final,vascular_log])

        with gr.Tab("08 · Visual Data Explorer"):
            gr.Markdown("## Explore the original experimental data before training or generation\nInspect the publisher reference image, segmentation, mapped absorption/scattering images, raw measured sinogram and selected time traces. This is a visual audit; it does not alter source files or calibrate the instrument.")
            with gr.Row():
                explorer_zip=gr.Textbox(value="data/external/openqpai/invision.zip",label="OpenQPAIData device ZIP")
                explorer_device=gr.Dropdown(["invision","tropus"],value="invision",label="Device")
                explorer_phantom=gr.Textbox(value="P.5.10.2",label="Phantom ID")
                explorer_wave=gr.Dropdown([700,750,800,850,900],value=700,label="Wavelength (nm)")
                explorer_out=gr.Textbox(value="experiments/dataset_visual_audit",label="Preview output folder")
            explorer_button=gr.Button("Load original dataset images and signal charts",variant="primary")
            explorer_gallery=gr.Gallery(label="Original dataset · reconstruction · segmentation · μa · μs′ · measured sinogram · measured traces",columns=3,height=680)
            explorer_report=gr.Code(language="json",label="Original acquisition metadata")
            explorer_button.click(_ui_dataset_visual_preview,[explorer_zip,explorer_device,explorer_phantom,explorer_wave,explorer_out],[explorer_gallery,explorer_report])
            with gr.Accordion("Independent optical measurements · DIS",open=False):
                dis_view_zip=gr.Textbox(value="data/external/openqpai/DIS.zip",label="DIS ZIP")
                dis_view_out=gr.Textbox(value="experiments/dis_visual_audit",label="DIS preview folder")
                dis_view_button=gr.Button("Plot DIS optical-property distributions")
                dis_view_gallery=gr.Gallery(label="DIS numeric distributions (native units)",columns=3,height=420)
                dis_view_report=gr.Code(language="json",label="DIS archive inventory")
                dis_view_button.click(_ui_dis_visual,[dis_view_zip,dis_view_out],[dis_view_gallery,dis_view_report])
            with gr.Accordion("SVOT volumetric reference viewer",open=False):
                svot_view_zip=gr.Textbox(value="data/external/openqpai/svot.zip",label="SVOT ZIP")
                svot_phantom=gr.Textbox(value="P.5.1",label="Phantom ID")
                svot_slice=gr.Slider(0,359,value=180,step=1,label="Volume slice index")
                svot_view_out=gr.Textbox(value="experiments/svot_visual_audit",label="SVOT preview folder")
                svot_view_button=gr.Button("Render SVOT volume slice")
                svot_view_gallery=gr.Gallery(label="SVOT labels · reconstructions · optical maps",columns=3,height=420)
                svot_view_report=gr.Code(language="json",label="SVOT slice metadata")
                svot_view_button.click(_ui_svot_visual,[svot_view_zip,svot_phantom,svot_slice,svot_view_out],[svot_view_gallery,svot_view_report])
        with gr.Tab("09 · Dataset Catalog & Source-Specific Tools"):
            gr.Markdown("""# Audited dataset catalog & source-specific research lanes
Each dataset has its own scientific purpose. A downloaded reference image is **not** a measured detector signal; skin-tone cohorts are not material-phantom masks; a simulated sinogram is not a human experimental acquisition. Registration and preview do not certify a reconstruction/training adapter.

**Publisher links you supplied:** Cambridge PATATO data ([dataset record](https://www.repository.cam.ac.uk/items/bf455f97-1ff0-4694-87ca-b72c85366dc6)) and Zenodo record 23348355 ([verify online](https://zenodo.org/records/23348355)). The second source's contents remain unverified; discovery will fail explicitly if the publisher record is inaccessible or absent. No MATLAB code is executed by the portal.""")
            from .dataset_registry import all_descriptions
            all_catalog=all_descriptions()
            gr.Dataframe(value=[[x['id'],x['title'],', '.join(x['lanes']),x['status']] for x in all_catalog],
                         headers=['ID','Dataset','Permitted scientific roles','Adapter status'],
                         interactive=False,label='Source-to-workflow compatibility matrix')
            with gr.Row():
                ds_info=gr.Dropdown([(x['title'],x['id']) for x in all_catalog],value='cambridge-patato',label='Dataset')
                ds_info_go=gr.Button('Show full scientific description')
            ds_info_text=gr.Code(language='json',label='Description / restrictions / publisher / license')
            ds_info_go.click(_ui_catalog_details,ds_info,ds_info_text)
            with gr.Accordion('Cambridge PATATO + user Zenodo record: destination-only download',open=True):
                gr.Markdown('The chosen directory is where **both** publishers deposit their assets; publisher discovery obtains the file list and size. License acceptance is required. A source can fail independently if unavailable. For individual budgets or selections use Tab 03 instead.')
                with gr.Row():
                    two_dest=gr.Textbox(value='data/external',label='Destination folder for both sources')
                    two_terms=gr.Checkbox(value=False,label='I reviewed both publisher licenses/terms')
                    two_button=gr.Button('Discover & download publisher assets',variant='primary')
                two_log=gr.Code(language='json',label='Individual publisher download statuses')
                two_button.click(_ui_two_links_download,[two_dest,two_terms],two_log)
            gr.Markdown('## Inspect a local dataset without loading the entire archive')
            with gr.Row():
                portal_source=gr.Dropdown([(x['title'],x['id']) for x in all_catalog],value='cambridge-patato',label='Source identity')
                portal_file=gr.Textbox(label='Local file, ZIP/TAR.XZ, directory or HDF5 path',value='data/external/cambridge-patato/preclinical_phantom.hdf5')
            portal_inspect=gr.Button('List files / datasets / arrays')
            portal_report=gr.Code(language='json',label='Schema and provenance inventory')
            portal_inspect.click(_ui_portal_inspect,[portal_source,portal_file],portal_report)
            with gr.Accordion('Preview a specifically selected numeric image or sinogram',open=True):
                with gr.Row():
                    portal_member=gr.Textbox(label='Optional member inside ZIP/TAR.XZ (exact case-sensitive path)',value='')
                    portal_key=gr.Textbox(label='HDF5/NPZ/MAT array key (e.g. raw_data or recons)',value='raw_data')
                    portal_index=gr.Number(value=0,precision=0,label='Sample / extra-axis index')
                    portal_semantic=gr.Dropdown(['image','signal'],value='signal',label='Actual data interpretation (publisher-defined)')
                portal_output=gr.Textbox(value='experiments/dataset_portal_previews',label='Preview output folder')
                portal_viz=gr.Button('Plot numeric data with provenance',variant='primary')
                portal_gallery=gr.Gallery(label='Original array charts (not synthetic generation)',columns=2,height=500)
                portal_viz_report=gr.Code(language='json',label='Array shape, source and assumptions')
                portal_viz.click(_ui_portal_visualize,[portal_source,portal_file,portal_key,portal_member,portal_index,portal_semantic,portal_output],
                                 [portal_gallery,portal_viz_report])
            with gr.Accordion('Synthetic Skin: extract paired optical maps + *simulated* signals',open=False):
                gr.Markdown('Only for HDF5 shards exposing the documented `mua`, `fluence`, `IP`, `sinogram` keys. No automatic inference of material labels or experimental provenance.')
                skin_h5=gr.Textbox(value='data/external/synthetic-skin/example_subset_100_samples.h5',label='Source HDF5 shard')
                skin_out=gr.Textbox(value='outputs/synthetic_skin_pairs',label='Output folder')
                skin_num=gr.Slider(1,100,value=8,step=1,label='Number of complete paired records')
                skin_go=gr.Button('Export true paired source arrays')
                skin_image=gr.Image(label='Optical/signal paired example')
                skin_report=gr.Code(language='json',label='Paired corpus manifest')
                skin_go.click(_ui_skin_pairs,[skin_h5,skin_out,skin_num],[skin_image,skin_report])
            with gr.Accordion('Cambridge PATATO: normalize actual detector signals + publisher reference',open=True):
                gr.Markdown('PATATO file format documents `raw_data` = [frames,wavelengths,detectors,time], `GEOMETRY` = [detectors,3] metres, root `fs` in Hz, `wavelengths`. This converter validates all of them; no fabricated detector locations.')
                pat_source=gr.Textbox(value='data/external/cambridge-patato/preclinical_phantom.hdf5',label='PATATO HDF5 file')
                with gr.Row():
                    pat_frame=gr.Number(value=0,precision=0,label='Frame index')
                    pat_wave=gr.Number(value=0,precision=0,label='Wavelength index (not nm)')
                    pat_group=gr.Textbox(value='PATATO_preclinical_phantom',label='Group ID for subject/phantom provenance')
                    pat_confirm=gr.Checkbox(value=False,label='Attest that this particular publisher scan was experimentally acquired')
                pat_out=gr.Textbox(value='data/real/patato_selected.h5',label='Native normalized HDF5 destination')
                pat_go=gr.Button('Normalize selected PATATO frame/wavelength for 2D reconstruction',variant='primary')
                pat_result=gr.Textbox(label='Normalized acquisition → Tab 02')
                pat_log=gr.Code(language='json',label='Detector geometry and acquisition metadata')
                pat_go.click(_ui_patato_normalize,[pat_source,pat_out,pat_frame,pat_wave,pat_group,pat_confirm],
                             [pat_result,pat_log]).then(lambda p:p,pat_result,real_path)
                pat_reference_key=gr.Textbox(value='recons/Reference Backprojection/0',label='Exact publisher reference reconstruction key')
                pat_reference_out=gr.Textbox(value='data/real/patato_reference.npy',label='Reference .npy destination')
                pat_ref_go=gr.Button('Extract publisher 2D reference (if present)')
                pat_ref_file=gr.Textbox(label='Reference path → Tab 02')
                pat_ref_log=gr.Code(language='json',label='Publisher reference details')
                pat_ref_go.click(_ui_patato_reference,[pat_source,pat_reference_out,pat_frame,pat_wave,pat_reference_key],
                                 [pat_ref_file,pat_ref_log]).then(lambda p:p,pat_ref_file,reference_path)

            with gr.Accordion('Advanced: normalize explicitly verified experimental HDF5 from other sources',open=False):
                gr.Markdown('For OADAT/PAISKINTONE/ShanghaiTech **only after** confirming HDF5 dataset shape, measured detector positions and sample rate from publisher metadata. This does not guess device axes or certify device calibration. Synthetic skin, ultrasound-CT images and vessel segmentation are blocked.')
                with gr.Row():
                    generic_key=gr.Textbox(value='raw_data',label='Raw signal HDF5 key')
                    generic_output=gr.Textbox(value='data/real/from_catalog.h5',label='Normalized output')
                    generic_pos=gr.Textbox(value='',label='Measured detector positions .npy [N,3] in metres')
                with gr.Row():
                    generic_det=gr.Number(value=0,precision=0,label='Detector axis')
                    generic_time=gr.Number(value=1,precision=0,label='Time axis')
                    generic_fixed=gr.Textbox(value='',label='Fixed other axes, e.g. 2=0,3=0')
                    generic_fs=gr.Number(value=40000000,label='Verified sample rate Hz')
                    generic_wave=gr.Number(value=700,label='Verified wavelength nm')
                generic_group=gr.Textbox(value='subject_or_phantom_group_1',label='Subject / phantom group ID')
                generic_confirm=gr.Checkbox(value=False,label='Attest experimental provenance and metadata from the original source')
                generic_go=gr.Button('Validate and normalize experimental HDF5')
                generic_ready=gr.Textbox(label='Normalized acquisition → Tab 02')
                generic_log=gr.Code(language='json',label='Import report')
                generic_go.click(_ui_generic_import,[portal_source,portal_file,generic_key,generic_output,generic_det,
                    generic_time,generic_fixed,generic_fs,generic_wave,generic_pos,generic_group,generic_confirm],
                    [generic_ready,generic_log]).then(lambda p:p,generic_ready,real_path)

            gr.Markdown('**Source-code packages and source datasets are different.** Any unverified Zenodo MATLAB code can be downloaded as an asset, but must be audited and licensed before its algorithms are adapted or executed. Generic HDF5/MAT inspection cannot replace verified device geometry/sampling metadata for experimental FISTA.')

        with gr.Tab("10 · Unified Reconstruction Lab"):
            gr.Markdown("""## One engine · synthetic and measured data · same algorithm registry

Choose a **PA-FORGE-native** `raw_signals.h5` (generated) or normalized measured acquisition.
Each method sees the exact same raw signals, detector positions, sample rate and grid.
Failures are reported **per method** rather than being silently replaced by another algorithm.
All comparisons are qualitative until device response, time zero and spatial orientation are calibrated.
Methods requiring a locally trained network **must** be given its `.pt` checkpoint.
""")
            with gr.Row():
                with gr.Column(scale=2):
                    recon_source=gr.Textbox(value='',label='Synthetic or measured normalized HDF5 source')
                    recon_uploaded=gr.File(label='Or upload native HDF5',file_types=['.h5','.hdf5'],type='filepath')
                    recon_choices=gr.Dropdown(
                        [(m['name']+' · '+m['category'],m['name']) for m in method_catalog()],
                        value=['das','ubp','cf_das','dmas','cg_tikhonov'],multiselect=True,
                        label='Choose 1–12 registered algorithms')
                    with gr.Row():
                        recon_grid=gr.Dropdown([48,64,96,128,192,256],value=64,label='2D image grid')
                        recon_fov=gr.Slider(5,200,value=100,step=5,label='FOV (mm), include sensors for FDTD')
                        recon_sound=gr.Slider(1000,2200,value=1500,step=10,label='Speed of sound (m/s)')
                    with gr.Row():
                        recon_iters=gr.Slider(1,80,value=8,step=1,label='Iterations')
                        recon_reg=gr.Number(value=.001,label='Regularization')
                        recon_dec=gr.Dropdown([1,2,4,8,16],value=4,label='FDTD anti-aliased time decimation')
                    recon_checkpoint=gr.Textbox(value='',label='Optional .pt checkpoint for learned approaches')
                    recon_truth=gr.Textbox(value='',label='Optional synthetic ground_truth.h5 (for matched-grid diagnostics only)')
                    recon_go=gr.Button('Compare selected reconstruction methods',variant='primary')
                with gr.Column(scale=3):
                    recon_gallery=gr.Gallery(label='Original signals · all reconstructed images',columns=3,height=620)
                    recon_report=gr.Code(label='Per-method runtime / caveats / diagnostic metrics',language='json')
                    recon_archive=gr.File(label='Download comparisons, numeric reconstructions, plots and manifest')
            recon_go.click(_ui_compare_shared,[recon_source,recon_uploaded,recon_choices,recon_grid,recon_fov,recon_sound,
                recon_iters,recon_reg,recon_dec,recon_checkpoint,recon_truth],[recon_gallery,recon_report,recon_archive])
            gr.Markdown('### Registered methods and scientific roles')
            recon_table=gr.Dataframe(value=[[x['name'],x['category'],x['model'],x['description']] for x in method_catalog()],
                headers=['Method','Family','Forward assumption','Description'],interactive=False)
            gr.Markdown('**PATATO methods** run through the optional original PATATO Python library, not a copied implementation. Its environment and dependencies are separate. **PATbox:** independent Python beamforming variants are not certified ports of the unavailable MATLAB source.')
        with gr.Tab("11 · Train Learned Reconstructors"):
            gr.Markdown("""## Supervised U-Net and residual CNN

Training requires **paired synthetic DAS reconstructions and synthetic initial-pressure ground truth**.
Source-group-disjoint train/validation split is mandatory. The application does **not**
train on experimental reference reconstructions as if they were calibrated p0 truth.
Model transfer to measured data is exploratory and must be evaluated on held-out devices/phantoms.
""")
            with gr.Row():
                with gr.Column(scale=2):
                    learned_root=gr.Textbox(value='outputs/paired_das_training',label='Root containing synthetic sample folders')
                    learned_weights=gr.Textbox(value='models/reconstruction_unet.pt',label='Checkpoint output')
                    learned_arch=gr.Dropdown(['unet','rescnn'],value='unet',label='Architecture')
                    learned_epochs=gr.Slider(1,150,value=10,step=1,label='Epochs')
                    learned_batch=gr.Dropdown([1,2,4,8,16],value=4,label='Batch size')
                    learned_seed=gr.Number(value=42,precision=0,label='Reproducibility seed')
                    learned_device=gr.Dropdown(['auto','cuda','cpu','mps'],value='auto',label='Training device')
                    learned_go=gr.Button('Train paired reconstruction model',variant='primary')
                with gr.Column(scale=3):
                    learned_result=gr.Textbox(label='Trained local checkpoint')
                    learned_log=gr.Code(language='json',label='Train/validation loss and group metadata')
            learned_go.click(_ui_train_reconstructor,[learned_root,learned_weights,learned_arch,learned_epochs,
                learned_batch,learned_seed,learned_device],[learned_result,learned_log]).then(
                lambda x:x,learned_result,recon_checkpoint)
        with gr.Tab("12 · Scientific guardrails & release") :
            gr.Markdown("""## PA-FORGE v0.9 — Shared Reconstruction Research Workspace

- **Synthetic:** Physically motivated 2D wave simulation, procedural / optional trained CVAE phantom inputs, selectable noise, UBP/DAS/adjoint/iterative reconstruction.
- **Experimental:** Real measured acquisition in normalized HDF5; actual detector coordinates; exploratory 2D UBP, DAS, filtered UBP, adjoint FDTD, or nonnegative FISTA.
- **Orientation:** Rotation and flips are **visual only**. The numerical reconstructions and measured signals remain unchanged. Use published device coordinate conventions for scientific registration.
- **No full 3D acoustics:** The simulation and experimental reconstruction are 2D, even if 3-component positions are stored.
- **No automatic real-image certification:** FOV, device response, sound speed and reference alignment need external verification.
- **Generative learning:** Separately train a categorical material CVAE on real phantom material labels, not human vasculature. Optical profile fitted on TRAIN groups only, with source units confirmed by the user. No pretrained clinically validated weights are bundled.\n- **Zenodo 23348355:** The user-supplied URL remains unverified as to record contents; no unsupported PATbox attribution or MATLAB translation claims. The other user-supplied Cambridge link is PATATO example **data**, not the Brown vascular dataset.
- **Local-only workflow:** Uploaded acquisition files are handled by the local Gradio app. Do not expose the local server publicly with sensitive data.

All new paths are labeled exploratory research functionality, not clinically validated. The original procedural and vascular modes remain available.
""")
        # Cross-tab workflow: training output paths are forwarded to the
        # experiment-informed generation tab without retyping.
        learn_prepare_event.then(lambda p:p,[optical_path],[gen_profile])
        train_event.then(lambda w:w,[trained_model],[gen_source])
    return demo
