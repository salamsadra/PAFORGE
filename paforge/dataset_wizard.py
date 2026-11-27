"""Goal-aware dataset selection, preflight planning and resumable full-corpus download."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import json, shutil
from .dataset_hub import discover, fetch_asset, SOURCE_BY_ID

RECOMMENDED_FILES = {
    # Optical/material learning needs mapped InVision labels + optical maps and
    # the independent DIS material measurements/mapping reference.
    'openqpai': {'DIS.zip','invision.zip','tropus.zip','material_mapping.json','README_data.md'},
    # Digital-twin QA uses the full data archive when the recommended profile is chosen.
    'digital-twin-iqa': {'data.zip','ipasc_file.hdf5','README.md'},
}
TASK_REQUIREMENTS = {
    'experiment_informed_generation': ['openqpai'],
    'full_research_corpus': ['openqpai','digital-twin-iqa'],
    'digital_twin_qa': ['digital-twin-iqa'],
    'custom': [],
}


def _asset_size(a):
    return int(a.get('bytes') or a.get('estimated_bytes') or 0)


def plan_datasets(task: str, selected_sources: list[str] | None, profile: str,
                  output_dir: str | Path, source_budgets_gb: dict[str,float] | None=None) -> dict:
    """Discover exact publisher assets and construct a storage-aware plan.

    profile: auto, starter, recommended, complete. 'complete' selects every
    public asset in every selected source. It does not extract huge archives.
    """
    if task not in TASK_REQUIREMENTS:
        raise ValueError(f'Unknown research objective: {task}')
    if profile not in ('auto','starter','recommended','complete'):
        raise ValueError('profile must be auto/starter/recommended/complete')
    root=Path(output_dir);root.mkdir(parents=True,exist_ok=True)
    mandatory=list(TASK_REQUIREMENTS[task])
    requested=list(dict.fromkeys((selected_sources or [])+mandatory))
    if not requested:
        raise ValueError('Select at least one dataset or research objective with mandatory sources')
    unknown=set(requested)-set(SOURCE_BY_ID)
    if unknown:raise ValueError(f'Unknown dataset IDs: {sorted(unknown)}')
    source_budgets_gb=source_budgets_gb or {}
    if not isinstance(source_budgets_gb,dict):raise ValueError('Expected source_id to positive GB mapping')
    for k,v in source_budgets_gb.items():
        if k not in requested:raise ValueError(f'Budget source {k!r} is not selected')
        if not isinstance(v,(int,float)) or not (0<float(v)<=10000):
            raise ValueError('Each dataset budget must be a positive number of GB')
    free=int(shutil.disk_usage(root).free)
    discovered={}; all_assets=[]; source_rows=[]
    for sid in requested:
        info=discover(sid);assets=info.get('assets',[])
        discovered[sid]=info
        if profile=='complete':
            chosen=assets
        elif profile=='recommended':
            names=RECOMMENDED_FILES.get(sid)
            chosen=[a for a in assets if a['filename'] in names] if names else [a for a in assets if _asset_size(a)<=1_000_000_000]
        elif profile=='starter':
            from .dataset_hub import pick_assets
            chosen=pick_assets(sid,assets,'starter')
        else:
            # Auto first evaluates the complete corpus; it is resolved below.
            chosen=assets
        source_rows.append({'id':sid,'title':info['source']['title'],'provider':info['source']['provider'],
                            'role':info['source']['role'],'mandatory':sid in mandatory,
                            'discovered_assets':len(assets),'selected_assets':len(chosen),
                            'access_note':info['source'].get('note',''),'landing_url':info['source'].get('landing_url')})
        for a in chosen:
            row={**a,'source_id':sid,'mandatory':(sid in mandatory and (sid!='openqpai' or a['filename'] in RECOMMENDED_FILES['openqpai']))}
            all_assets.append(row)
    complete_bytes=sum(_asset_size(a) for a in all_assets)
    # Auto mode: full only when complete assets + reserve fit; otherwise
    # recommended, then starter. We recompute actual assets for the chosen mode.
    resolved=profile
    if profile=='auto':
        reserve=2_000_000_000
        if complete_bytes+reserve<=free:
            resolved='complete'
        else:
            all_assets=[];resolved='recommended'
            for sid in requested:
                assets=discovered[sid].get('assets',[]);names=RECOMMENDED_FILES.get(sid)
                chosen=[a for a in assets if a['filename'] in names] if names else [a for a in assets if _asset_size(a)<=1_000_000_000]
                for a in chosen:all_assets.append({**a,'source_id':sid,'mandatory':sid in mandatory})
            rec_bytes=sum(_asset_size(a) for a in all_assets)
            if rec_bytes+reserve>free:
                all_assets=[];resolved='starter'
                from .dataset_hub import pick_assets
                for sid in requested:
                    for a in pick_assets(sid,discovered[sid].get('assets',[]),'starter'):
                        all_assets.append({**a,'source_id':sid,'mandatory':sid in mandatory})
    # Exact per-source caps select WHOLE publisher assets; never truncate a ZIP/HDF5.
    # A budget which excludes the required optical/material inputs fails loudly.
    if source_budgets_gb:
        limited=[]
        for sid in requested:
            options=[a for a in all_assets if a['source_id']==sid]
            cap=source_budgets_gb.get(sid)
            if cap is None:
                limited.extend(options);continue
            allowed=int(cap*1e9)
            essentials=set()
            if sid=='openqpai' and task in ('experiment_informed_generation','full_research_corpus'):
                essentials={'invision.zip','DIS.zip','material_mapping.json'}
            missing=essentials-set(a['filename'] for a in options)
            if missing:raise ValueError('Optical learning requires missing files: '+', '.join(sorted(missing)))
            required=[a for a in options if a['filename'] in essentials]
            used=sum(_asset_size(a) for a in required)
            if used>allowed:
                raise ValueError(f'Budget for {sid} ({cap:g} GB) is too small for mandatory files ({used/1e9:.2f} GB). Increase limit; archives cannot be partially downloaded.')
            # Small files first, then larger archives, deterministic ties.
            optional=sorted((a for a in options if a['filename'] not in essentials),
                             key=lambda a:(_asset_size(a),a['filename']))
            limited.extend(required)
            for a in optional:
                if used+_asset_size(a)<=allowed:
                    limited.append(a);used+=_asset_size(a)
        all_assets=limited
    # Keep source counts aligned with the final resolved profile (especially auto).
    for source_row in source_rows:
        source_row['selected_assets']=sum(1 for a in all_assets if a['source_id']==source_row['id'])
        source_row['resolved_profile']=resolved
    # Existing files are not charged to the remaining download budget.
    assets_out=[];remaining=0;existing=0
    for a in all_assets:
        sid=a['source_id'];dest=root/sid/a['filename'];size=_asset_size(a)
        exact=int(a.get('bytes') or 0);estimated=int(a.get('estimated_bytes') or 0)
        present=dest.is_file() and (not exact or dest.stat().st_size==exact)
        if present:existing+=size
        else:remaining+=size
        assets_out.append({'source_id':sid,'filename':a['filename'],'bytes':size,
                           'publisher_bytes':exact,'estimated_bytes':estimated,
                           'size_gb':round(size/1e9,4),'mandatory':bool(a.get('mandatory')),
                           'already_present_size_checked':present,'checksum':a.get('checksum'),
                           'url':a.get('url'),'checksum_algorithm':a.get('checksum_algorithm')})
    reserve=2_000_000_000
    plan={'project':'PA-FORGE','wizard_version':'0.8.0','created_utc':datetime.now(timezone.utc).isoformat(),
            'task':task,'requested_sources':requested,'mandatory_sources':mandatory,
            'profile_requested':profile,'profile_resolved':resolved,'source_budgets_gb':source_budgets_gb,'output_dir':str(root),
            'sources':source_rows,'assets':assets_out,
            'totals':{'asset_count':len(assets_out),'complete_selected_bytes':sum(x['bytes'] for x in assets_out),
                      'already_present_bytes':existing,'remaining_download_bytes':remaining,
                      'remaining_download_gb':round(remaining/1e9,3),'free_disk_bytes':free,
                      'free_disk_gb':round(free/1e9,2),'reserve_bytes':reserve,
                      'preflight_ok':free>=remaining+reserve,
                      'note':'Per-source budgets select whole publisher files, never partial archive bytes. Extraction can require substantially more disk.'},
            'scientific_roles':{
                'DIS.zip':'Independent double-integrating-sphere optical material measurements; retain for optical-property audit/calibration.',
                'invision.zip':'MSOT InVision raw IPASC signals, 2D reconstructions, manual labels, mapped mua/musp images.',
                'tropus.zip':'TROPUS raw IPASC signals, 2D reconstructions, labels and mapped optical maps.',
                'svot.zip':'SVOT 3D reconstructions, labels and optical volumes; publisher does not share SVOT raw signals.',
                'data.zip':'Digital-twin full-reference QA dataset; keep separate from OpenQPAIData training split.'}}
    plan_path=root/'dataset_plan.json'
    plan['plan_path']=str(plan_path)
    plan_path.write_text(json.dumps(plan,indent=2),encoding='utf-8')
    return plan


def execute_dataset_plan(plan: dict, accept_licenses: bool, progress=None) -> dict:
    if not accept_licenses:
        raise ValueError('Review every publisher license and confirm before downloading.')
    totals=plan['totals'];root=Path(plan['output_dir']);root.mkdir(parents=True,exist_ok=True)
    # Recalculate at execution time: the user may have used disk space after planning.
    free_now=shutil.disk_usage(root).free
    remaining_now=0
    for row in plan['assets']:
        dest=root/row['source_id']/row['filename']
        exact=int(row.get('publisher_bytes') or 0);estimate=int(row.get('estimated_bytes') or row.get('bytes') or 0)
        if not (dest.is_file() and (not exact or dest.stat().st_size==exact)):
            remaining_now+=estimate
    if free_now < remaining_now + int(totals.get('reserve_bytes',2_000_000_000)):
        raise RuntimeError(f"Insufficient disk space at download time. Need {remaining_now/1e9:.2f} GB plus 2 GB reserve; free {free_now/1e9:.2f} GB.")
    result={'project':'PA-FORGE','version':'0.7.0','task':plan['task'],'profile':plan['profile_resolved'],
            'sources':{},'downloaded_bytes':0,'already_present_bytes':totals['already_present_bytes'],
            'notice':'Download does not imply ingestion, calibration or validation.'}
    sizes=[a['bytes'] for a in plan['assets']];grand=max(1,sum(sizes));done=0
    for row in plan['assets']:
        sid=row['source_id'];result['sources'].setdefault(sid,{'assets':[],'status':'running'})
        asset={'filename':row['filename'],'bytes':row.get('publisher_bytes',row['bytes']),
               'estimated_bytes':row.get('estimated_bytes',0),'url':row['url'],
               'checksum_algorithm':row['checksum_algorithm'],'checksum':row['checksum']}
        def cb(filename,received,total):
            if progress:
                progress(min(.99,(done+received)/grand),desc=f'{sid}: {filename} ({received/1e9:.2f}/{max(total,1)/1e9:.2f} GB)')
        try:
            got=fetch_asset(asset,root/sid,min_free_gb=.5,
                            max_download_bytes=(row.get('publisher_bytes') or None),progress_callback=cb)
            result['sources'][sid]['assets'].append(got)
            if got['status']=='downloaded':result['downloaded_bytes']+=got['bytes']
            done+=row['bytes']
        except Exception as exc:
            result['sources'][sid]['assets'].append({'filename':row['filename'],'status':'error','error':str(exc)})
            result['sources'][sid]['status']='partial'
            # Continue other files so a single bad publisher asset does not
            # erase completed, checksum-verified downloads.
            done+=row['bytes']
            continue
    for sid,source in result['sources'].items():
        if source['status']!='partial':
            states=[a['status'] for a in source['assets']]
            source['status']='complete_selected' if all(s in ('downloaded','already_verified','already_exists_size_checked') for s in states) else 'partial'
    result['report_path']=str(root/'dataset_wizard_report.json')
    (root/'dataset_wizard_report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    if progress:progress(1.0,desc='Dataset download plan finished')
    return result


def inspect_downloaded_archives(root: str | Path, output_json: str | Path | None = None) -> dict:
    """Inspect ZIP central directories without extracting large research archives."""
    from zipfile import ZipFile, BadZipFile
    from collections import Counter
    root=Path(root);report={'root':str(root),'archives':[]}
    for path in sorted(root.rglob('*.zip')):
        item={'path':str(path),'compressed_bytes':path.stat().st_size}
        try:
            with ZipFile(path) as z:
                infos=z.infolist();ext=Counter();roles=Counter();top=Counter();uncompressed=0
                for info in infos:
                    if info.is_dir():continue
                    uncompressed+=info.file_size
                    suffix=Path(info.filename).suffix.lower() or '[no extension]';ext[suffix]+=1
                    parts=Path(info.filename).parts
                    if parts:top[parts[0]]+=1
                    low=info.filename.lower()
                    for role in ('raw','recon','labels','mua','musp','processed','manual'):
                        if f'/{role}/' in low:roles[role]+=1
                item.update({'status':'readable','file_count':len([x for x in infos if not x.is_dir()]),
                    'uncompressed_bytes':uncompressed,'uncompressed_gb':round(uncompressed/1e9,3),
                    'extension_counts':dict(ext),'role_counts':dict(roles),'top_level_counts':dict(top),
                    'sample_paths':[x.filename for x in infos if not x.is_dir()][:30]})
        except (BadZipFile,OSError) as exc:item.update({'status':'error','error':str(exc)})
        report['archives'].append(item)
    report['archive_count']=len(report['archives'])
    if output_json:
        p=Path(output_json);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report
