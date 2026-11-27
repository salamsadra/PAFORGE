"""Publisher-aware, resumable dataset acquisition for PA-FORGE.

Download does NOT imply format harmonization or an experimentally calibrated model.
Each acquisition retains a per-asset provenance manifest and verified checksum when
published upstream. This module uses only the Python standard library.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import ssl
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

USER_AGENT = 'PA-FORGE-Dataset-Hub/0.8 (academic research; publisher metadata APIs)'


@dataclass(frozen=True)
class Source:
    id: str
    title: str
    provider: str
    record: str
    role: str
    license_note: str
    anatomy: tuple[str, ...]
    note: str = ''
    zenodo_id: int | None = None
    dryad_doi: str | None = None
    landing_url: str | None = None


SOURCES = (
    Source('openqpai', 'OpenQPAIData', 'zenodo', 'https://zenodo.org/records/14044853',
           'experimental phantom raw signals / reference labels', 'CC BY 4.0 (check publisher record)',
           ('phantom', 'skin'), zenodo_id=14044853),
    Source('digital-twin-iqa', 'Digital twins for full-reference PA reconstruction QA', 'zenodo',
           'https://zenodo.org/records/15388429',
           'small IPASC HDF5 acquisition and full 2.9 GB digital-twin benchmark',
           'CC BY 4.0 (verify publisher record and credit original authors)',
           ('phantom',), zenodo_id=15388429),
    Source('paiskintone', 'PAISKINTONE', 'zenodo', 'https://zenodo.org/records/20068219',
           'human multispectral raw acquisitions', 'Verify the current publisher license before use',
           ('forearm', 'skin', 'carotid', 'vascular'), zenodo_id=20068219),
    Source('synthetic-skin', 'Synthetic Vascularized Digital Skin Phantoms', 'zenodo',
           'https://zenodo.org/records/21785486', 'synthetic tissue/pressure/sinogram maps',
           'Verify the current publisher license before use', ('skin', 'vascular'), zenodo_id=21785486),
    Source('breast-pact', 'Breast PACT/URCT (Lin et al.)', 'dryad',
           'https://doi.org/10.5061/dryad.vmcvdnd55', 'experimental reconstructed breast images; raw not confirmed',
           'Verify the current publisher/Dryad terms', ('breast',), dryad_doi='10.5061/dryad.vmcvdnd55'),
    Source('pam3', 'PAM3 breast tomography reconstructions', 'zenodo',
           'https://zenodo.org/records/20612980', 'reconstructed volume references; raw not confirmed',
           'Check publisher record and individual asset terms', ('breast',), zenodo_id=20612980),
    Source('photiqa', 'PhotIQA', 'zenodo', 'https://zenodo.org/records/15828190',
           'reconstructed image quality ratings', 'CC BY 4.0 (check publisher record)',
           ('phantom',), zenodo_id=15828190),
    Source('shanghaitech-vessel', 'ShanghaiTech vessel-phantom photoacoustic dataset', 'zenodo',
           'https://zenodo.org/records/10719987', 'vascular data; raw availability requires inspection',
           'Verify current upstream license and reuse restrictions', ('vascular',), zenodo_id=10719987),
    Source('oadat', 'OADAT', 'guided', 'https://www.research-collection.ethz.ch/handle/20.500.11850/551512',
           'experimental human forearm raw signals and synthetic phantoms', 'CC BY-NC 4.0 (noncommercial)',
           ('forearm', 'vascular'),
           note='ETH Libdrive public share (~842 GB aggregate); choose files via browser. Bulk URL unverified.',
           landing_url='https://libdrive.ethz.ch/index.php/s/i974KegIjhJJxkK'),
    Source('cambridge-patato', 'Cambridge PATATO example acquisitions', 'cambridge_api',
           'https://www.repository.cam.ac.uk/items/bf455f97-1ff0-4694-87ca-b72c85366dc6',
           'real phantom and mouse experimental raw_data/recons HDF5',
           'CC BY 4.0 (check Cambridge repository)', ('phantom','mouse'),
           note='Download each asset through publisher DSpace metadata; no guessed bitstream URLs',
           landing_url='https://www.repository.cam.ac.uk/items/bf455f97-1ff0-4694-87ca-b72c85366dc6'),
    Source('zenodo-23348355', 'Unverified user-supplied Zenodo record 23348355', 'zenodo',
           'https://zenodo.org/records/23348355',
           'unknown: files and schema not verified',
           'License not independently verified', ('unknown',), zenodo_id=23348355,
           note='No assertion this contains PATbox or MATLAB code until publisher metadata is retrieved'),
    Source('cambridge-vascular', 'Cambridge vascular-network mesoscopy', 'cambridge',
           'https://doi.org/10.17863/CAM.78208', 'vascular anatomy/reconstructed images; raw not confirmed',
           'CC BY-NC 4.0 reported; check record', ('vascular', 'skin'),
           note='Publisher-hosted static bitstream links; file-size approximations used for download budgets.',
           landing_url='https://www.repository.cam.ac.uk/items/ba93a2ad-7957-42c6-ba81-f6e9c1af5609'),
)
SOURCE_BY_ID = {s.id: s for s in SOURCES}

# Public bitstream URLs were taken from Cambridge's dataset landing page (2026-10-28).
# Displayed sizes are approximate, so DO NOT assert exact byte counts/checksums.
CAMBRIDGE_ASSETS = (
    ('InSilico.zip','f94b4c03-8fb5-4e61-9007-d122c078cf79',433_040_000),
    ('InVivo.zip','f54eaa1a-f5e6-406f-b40d-cd55287172dd',1_680_000_000),
    ('Phantom.zip','7be13a3b-27aa-401e-a303-dcc207795e97',1_170_000_000),
    ('3Dunet_rsom.hdf5','e61743fc-9386-4243-af28-0ea0e5aa460f',259_350_000),
)


def _source_cambridge(source: Source) -> list[dict]:
    return [dict(filename=n, bytes=0, estimated_bytes=size,
                 url=f'https://www.repository.cam.ac.uk/bitstreams/{uid}/download',
                 checksum_algorithm=None, checksum=None, upstream_file_id=uid)
            for n,uid,size in CAMBRIDGE_ASSETS]



class SourceError(RuntimeError):
    """Publication metadata could not be retrieved or parsed."""


class DownloadError(RuntimeError):
    """Download or checksum verification failed."""


def _https(url: str) -> str:
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != 'https' or not parts.hostname or parts.username or parts.password:
        raise SourceError(f'Unsafe/non-HTTPS asset URL: {url!r}')
    return url


def _open(url: str, headers: dict[str, str] | None = None, timeout: int = 40):
    req = urllib.request.Request(_https(url), headers={'User-Agent': USER_AGENT, **(headers or {})})
    return urllib.request.urlopen(req, timeout=timeout)


def _json_url(url: str) -> dict:
    with _open(url, {'Accept': 'application/json'}, timeout=35) as stream:
        if getattr(stream, 'status', 200) >= 400:
            raise SourceError(f'HTTP {stream.status}: {url}')
        content = stream.read(15 * 1024 * 1024 + 1)
        if len(content) > 15 * 1024 * 1024:
            raise SourceError('Publisher metadata unexpectedly large')
        return json.loads(content)


def _basename(s: str) -> str:
    # Prevent remote names from escaping the source-specific download directory.
    if not s or s in ('.', '..') or '\\' in s or '/' in s or '\x00' in s or s.startswith('.'):
        raise SourceError(f'Unsafe publisher filename {s!r}')
    return s


def _checksum(raw: str | None) -> tuple[str | None, str | None]:
    if not raw:
        return None, None
    raw = raw.lower()
    if ':' in raw:
        alg, val = raw.split(':', 1)
    elif re.fullmatch(r'[0-9a-f]{32}', raw):
        alg, val = 'md5', raw
    elif re.fullmatch(r'[0-9a-f]{64}', raw):
        alg, val = 'sha256', raw
    else:
        return None, None
    if alg not in ('md5', 'sha256') or not re.fullmatch(r'[0-9a-f]+', val):
        return None, None
    if len(val) != {'md5': 32, 'sha256': 64}[alg]:
        return None, None
    return alg, val


def _source_zenodo(source: Source) -> list[dict]:
    data = _json_url(f'https://zenodo.org/api/records/{source.zenodo_id}')
    assets = data.get('files') or []
    if isinstance(assets, dict):
        assets = list(assets.values())
    out = []
    for a in assets:
        filename = _basename(a.get('key') or a.get('filename') or '')
        links = a.get('links') or {}
        url = links.get('content') or links.get('self')
        if not url:
            # Official Zenodo records API, not a guessed third-party CDN.
            url = f'https://zenodo.org/records/{source.zenodo_id}/files/{urllib.parse.quote(filename)}?download=1'
        alg, digest = _checksum(a.get('checksum') or a.get('md5'))
        out.append({'filename': filename, 'bytes': int(a.get('size') or a.get('filesize') or 0),
                    'url': _https(url), 'checksum_algorithm': alg, 'checksum': digest,
                    'upstream_file_id': a.get('id')})
    if not out:
        raise SourceError(f'No published Zenodo assets found for {source.id}')
    return out


def _dryad_links(obj: dict) -> str | None:
    links = obj.get('_links') or {}
    for name in ('stash:download', 'download', 'stash:files', 'self'):
        x = links.get(name)
        if isinstance(x, dict) and x.get('href'):
            return x['href']
    return None


def _source_dryad(source: Source) -> list[dict]:
    doi = urllib.parse.quote('doi:' + source.dryad_doi, safe='')
    base = 'https://datadryad.org/api/v2'
    data = _json_url(f'{base}/datasets/{doi}/versions')
    versions = data.get('_embedded', {}).get('stash:versions') or data.get('versions') or []
    if not versions and isinstance(data.get('_embedded'), dict):
        for value in data['_embedded'].values():
            if isinstance(value, list) and value and isinstance(value[0], dict) and 'id' in value[0]:
                versions = value
                break
    if not versions:
        raise SourceError('Dryad versions unavailable; use the publisher link to inspect access')
    # Latest published version should be last, unless versionNumber supplied.
    version = max(versions, key=lambda v: int(v.get('versionNumber') or v.get('id') or 0))
    version_id = version['id']
    assets = []
    next_url = f'{base}/versions/{version_id}/files?per_page=100'
    for _ in range(20):
        response = _json_url(next_url)
        chunk = response.get('_embedded', {}).get('stash:files') or response.get('files') or []
        if not chunk and isinstance(response.get('_embedded'), dict):
            for v in response['_embedded'].values():
                if isinstance(v, list):
                    chunk = v
                    break
        for a in chunk:
            name = _basename((a.get('path') or a.get('filename') or '').rsplit('/', 1)[-1])
            fid = a.get('id')
            url = f'{base}/files/{fid}/download' if fid is not None else _dryad_links(a)
            if not url:
                raise SourceError(f'Dryad downloadable URL missing for {name}')
            alg, checksum = _checksum(
                f"{a['digestType']}:{a['digest']}" if a.get('digestType') and a.get('digest') else None)
            assets.append({'filename': name, 'bytes': int(a.get('size') or 0),
                           'url': _https(url), 'checksum_algorithm': alg, 'checksum': checksum,
                           'upstream_file_id': fid})
        more = response.get('_links', {}).get('next', {}).get('href')
        if not more:
            break
        next_url = urllib.parse.urljoin(base, more)
    if not assets:
        raise SourceError('Dryad version has no public asset metadata')
    return assets


def _source_cambridge_api(source: Source) -> list[dict]:
    """Resolve *real* publisher bitstreams from DSpace REST, never guess UUIDs.

    Supports DSpace 7/8 HAL lists with _embedded.bitstreams / _embedded.bundles.
    Publisher HTML landing pages have no stable download filename mapping.
    """
    import urllib.parse
    ident='bf455f97-1ff0-4694-87ca-b72c85366dc6'
    base='https://www.repository.cam.ac.uk/server/api/core'
    def permitted(url):
        parsed=urllib.parse.urlparse(url)
        if parsed.scheme!='https' or parsed.hostname!='www.repository.cam.ac.uk':
            raise SourceError('Cambridge REST returned external asset URL; publisher review required')
        return url
    def collect(url, embedded_key, limit=8):
        results=[]
        for _ in range(limit):
            data=_json_url(permitted(url))
            results.extend(data.get('_embedded',{}).get(embedded_key,[]))
            next_page=data.get('_links',{}).get('next')
            if not next_page or not next_page.get('href'):break
            url=permitted(next_page['href'])
        return results
    try:
        bundles=collect(f'{base}/items/{ident}/bundles?size=100','bundles')
        assets=[];seen=set()
        for bundle in bundles:
            name=str(bundle.get('name',''))
            if name not in ('ORIGINAL','CONTENT','DATA','THUMBNAIL','LICENSE'):
                continue
            if name in ('THUMBNAIL','LICENSE'):
                continue
            href=(bundle.get('_links',{}).get('bitstreams') or {}).get('href')
            if not href:continue
            for b in collect(href+'?size=100' if '?' not in href else href,'bitstreams'):
                fname=_basename(str(b.get('name') or ''))
                if fname in seen:continue
                url=(b.get('_links',{}).get('content') or {}).get('href')
                if not url:continue
                url=permitted(url)
                size=int(b.get('sizeBytes') or b.get('size') or 0)
                if size<0:continue
                assets.append(dict(filename=fname,bytes=size,url=url,
                                   checksum_algorithm=None,checksum=None,upstream_file_id=b.get('uuid')))
                seen.add(fname)
        if not assets:
            raise SourceError('Cambridge DSpace REST returned no original bitstreams; use publisher page / local files')
        return assets
    except (KeyError,TypeError,ValueError,urllib.error.URLError,SourceError) as exc:
        raise SourceError('Cambridge PATATO asset discovery unavailable from DSpace REST. '
             'The local-file inspector still works after manual publisher download. '
             'Reason: '+str(exc)) from exc


def discover(source_id: str) -> dict:
    if source_id not in SOURCE_BY_ID:
        raise KeyError(f'Unknown dataset {source_id}; choices: {", ".join(SOURCE_BY_ID)}')
    s = SOURCE_BY_ID[source_id]
    info = {'source': asdict(s), 'assets': [], 'state': 'guided' if s.provider == 'guided' else 'discoverable'}
    if s.provider == 'zenodo':
        info['assets'] = _source_zenodo(s)
    elif s.provider == 'dryad':
        info['assets'] = _source_dryad(s)
    elif s.provider == 'cambridge':
        info['assets'] = _source_cambridge(s)
    elif s.provider == 'cambridge_api':
        info['assets'] = _source_cambridge_api(s)
    return info


def pick_assets(source_id: str, assets: list[dict], profile: str) -> list[dict]:
    if profile not in ('starter', 'full'):
        raise ValueError('profile must be starter or full')
    if profile == 'full':
        return sorted(assets, key=lambda a: a['filename'].lower())
    filenames = {
        'openqpai': ('invision.zip', 'DIS.zip', 'material_mapping.json', 'README_data.md'),
        'digital-twin-iqa': ('ipasc_file.hdf5','README.md'),
        'paiskintone': ('PAISKINTONE-1.0.0.zip', 'SKIN42.tar.xz'),
        'synthetic-skin': ('example_subset_100_samples.h5', 'README.md', 'requirements.txt'),
        'breast-pact': ('README.md',),
        'photiqa': ('annotations.zip', 'normalised_images.zip', 'data_descriptor.md'),
        'cambridge-vascular': ('InSilico.zip',),
        'cambridge-patato': ('README','preclinical_phantom.hdf5'),
        'zenodo-23348355': (),
    }
    if source_id in filenames:
        allowed = set(filenames[source_id]); return [a for a in assets if a['filename'] in allowed]
    # Non-pinned Zenodo archives: choose descriptions + smallest nontrivial data
    docs = [a for a in assets if a['filename'].lower().endswith(('.md', '.txt', '.json', '.csv')) and a['bytes'] < 20_000_000]
    data = [a for a in assets if a not in docs and a['filename'].lower().endswith(('.zip', '.mat', '.h5', '.hdf5', '.npz', '.npy', '.tar.gz', '.tar.xz'))]
    feasible = [a for a in data if 0 < a['bytes'] <= 1_000_000_000]
    return (docs[:4] + sorted(feasible, key=lambda a: a['bytes'])[:1])


def _file_hash(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fetch_asset(asset: dict, folder: Path, *, dry_run: bool = False,
                min_free_gb: float = 0.25, max_download_bytes: int | None = None,
                progress_callback=None) -> dict:
    """Fetch one asset: resume partials, verify upstream checksum, atomic rename.

    If Range is ignored, restart rather than appending duplicate bytes. An upstream
    server that rejects downloads fails explicitly and is not counted as fetched.
    """
    folder = Path(folder)
    filename = _basename(asset['filename'])
    url = _https(asset['url'])
    expected = int(asset.get('bytes') or 0)
    alg, digest = asset.get('checksum_algorithm'), asset.get('checksum')
    final = folder / filename
    partial = folder / (filename + '.part')
    if dry_run:
        return {'filename': filename, 'status': 'planned', 'bytes': expected}
    folder.mkdir(parents=True, exist_ok=True)
    if final.exists():
        if expected and final.stat().st_size != expected:
            raise DownloadError(f'Existing file has unexpected size: {final}')
        if digest and _file_hash(final, alg) != digest:
            raise DownloadError(f'Existing file has incorrect {alg}: {final}')
        return {'filename': filename, 'status': 'already_verified' if digest else 'already_exists_size_checked',
                'bytes': final.stat().st_size, 'checksum_verified': bool(digest), 'path': str(final)}
    existing = partial.stat().st_size if partial.exists() else 0
    estimated = int(asset.get('estimated_bytes') or 0)
    needed = max(0, (expected or estimated) - existing)
    free = shutil.disk_usage(folder).free
    if expected and needed + int(min_free_gb * 1e9) > free:
        raise DownloadError(f'Insufficient free disk for {filename}: need at least {needed/1e9:.2f} GB plus reserve')
    headers = {'Range': f'bytes={existing}-'} if existing else {}
    try:
        with _open(url, headers, timeout=90) as stream:
            status = getattr(stream, 'status', 200)
            if status == 416 and expected == existing:
                pass
            elif status not in (200, 206):
                raise DownloadError(f'Unexpected HTTP {status} for {filename}')
            else:
                content_range = stream.headers.get('Content-Range', '')
                resume = existing > 0 and status == 206 and content_range.startswith(f'bytes {existing}-')
                if status == 206 and not resume:
                    raise DownloadError(f'Unexpected partial response range for {filename}')
                with partial.open('ab' if resume else 'wb') as out:
                    downloaded = existing if resume else 0
                    total_hint = expected or estimated or int(stream.headers.get('Content-Length') or 0) + existing
                    for chunk in iter(lambda: stream.read(2 * 1024 * 1024), b''):
                        downloaded += len(chunk)
                        if max_download_bytes is not None and downloaded > max_download_bytes:
                            raise DownloadError('Asset exceeds --max-file-gb or --max-total-gb while streaming')
                        out.write(chunk)
                        if progress_callback is not None:
                            progress_callback(filename, downloaded, total_hint)
    except Exception as exc:
        # Keep .part for recovery, never claim success.
        raise DownloadError(f'{filename}: {type(exc).__name__}: {exc}') from exc
    if not partial.exists():
        raise DownloadError(f'{filename}: no archive created')
    size = partial.stat().st_size
    if expected and size != expected:
        raise DownloadError(f'{filename}: incomplete transfer ({size} of {expected} bytes); re-run to resume')
    if digest and _file_hash(partial, alg) != digest:
        partial.unlink(missing_ok=True)  # corrupted bytes must not be resumed
        raise DownloadError(f'{filename}: upstream {alg} checksum mismatch; removed corrupt partial')
    # In particular, publisher-hosted bitstreams without a published checksum
    # must not be mistaken for data if the server returns a 200 HTML login page.
    if not digest:
        import zipfile
        ext = filename.lower()
        with partial.open('rb') as check:
            magic = check.read(16)
        if ext.endswith('.zip') and not zipfile.is_zipfile(partial):
            partial.unlink(missing_ok=True)
            raise DownloadError(f'{filename}: received bytes are not a valid ZIP (perhaps an HTML login page)')
        if ext.endswith(('.h5', '.hdf5')) and magic[:8] != b'\x89HDF\r\n\x1a\n':
            partial.unlink(missing_ok=True)
            raise DownloadError(f'{filename}: not a valid HDF5 signature')
        if ext.endswith('.tar.xz') and magic[:6] != b'\xfd7zXZ\x00':
            partial.unlink(missing_ok=True)
            raise DownloadError(f'{filename}: not a valid XZ signature')
    partial.replace(final)
    return {'filename': filename, 'status': 'downloaded', 'bytes': size,
            'checksum_verified': bool(digest), 'checksum_algorithm': alg, 'path': str(final)}


def safe_unpack(archive: Path, target: Path, max_expanded_gb: float = 100.0) -> dict:
    """Optional archive extraction with path/symlink/size protection."""
    import tarfile
    import zipfile
    archive = Path(archive)
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    cap = int(max_expanded_gb * 1e9)
    def _good(name):
        dest = (target / name).resolve()
        if not dest.is_relative_to(target.resolve()):
            raise DownloadError(f'Unsafe path in archive: {name}')
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            info = z.infolist()
            if sum(i.file_size for i in info) > cap:
                raise DownloadError('Expanded ZIP exceeds --max-expanded-gb')
            for i in info:
                _good(i.filename)
                if (i.external_attr >> 16) & 0o170000 == 0o120000:
                    raise DownloadError('Archive contains a symlink')
            z.extractall(target)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as t:
            members = t.getmembers()
            if sum(m.size for m in members) > cap:
                raise DownloadError('Expanded TAR exceeds --max-expanded-gb')
            for m in members:
                _good(m.name)
                if not (m.isfile() or m.isdir()):
                    raise DownloadError('Archive contains special file/link')
            for m in members:
                if m.isdir():
                    (target / m.name).mkdir(parents=True, exist_ok=True)
                elif m.isfile():
                    location = target / m.name
                    location.parent.mkdir(parents=True, exist_ok=True)
                    source = t.extractfile(m)
                    if source is None:
                        raise DownloadError('Cannot extract file in TAR')
                    with source, location.open('wb') as out:
                        shutil.copyfileobj(source, out, 2 * 1024 * 1024)
    else:
        raise DownloadError(f'Unsupported extraction format: {archive.name}')
    return {'archive': str(archive), 'extracted_to': str(target)}


def run_download(source_ids: list[str] | None, root: Path, *, profile: str = 'starter',
                 accept_licenses: bool = False, max_file_gb: float = 2.0,
                 max_total_gb: float = 5.0, unpack: bool = False,
                 dry_run: bool = False) -> dict:
    """Generate a complete machine-readable status report, even on per-source errors."""
    if profile not in ('starter', 'full'):
        raise ValueError('Only starter and full profiles supported')
    if not dry_run and not accept_licenses:
        raise ValueError('Explicit --accept-licenses required before downloading third-party datasets')
    if max_file_gb <= 0 or max_total_gb <= 0:
        raise ValueError('Positive disk budgets required')
    root = Path(root)
    ids = list(dict.fromkeys(source_ids if source_ids is not None else SOURCE_BY_ID.keys()))
    result: dict[str, Any] = {
        'project': 'PA-FORGE', 'version': '0.7.0', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'requested': ids, 'profile': profile, 'dry_run': dry_run,
        'sources': {}, 'notice': 'Downloading is NOT ingestion, physics calibration, or independent experimental validation.'}
    aggregate = 0
    for sid in ids:
        if sid not in SOURCE_BY_ID:
            result['sources'][sid] = {'status': 'error', 'error': 'Unknown dataset ID'}
            continue
        source = SOURCE_BY_ID[sid]
        output: dict[str, Any] = {'title': source.title, 'provider': source.provider,
                                  'role': source.role, 'license': source.license_note,
                                  'publisher': source.record, 'assets': []}
        result['sources'][sid] = output
        if source.provider == 'guided':
            output.update({'status': 'guided_external_download', 'reason': source.note,
                           'access_url': source.landing_url or source.record})
            continue
        try:
            inventory = discover(sid)
            chosen = pick_assets(sid, inventory['assets'], profile)
            output['discovered_files'] = len(inventory['assets'])
            output['selected_files'] = len(chosen)
            output['available_bytes_estimate'] = sum(a.get('bytes') or a.get('estimated_bytes') or 0 for a in inventory['assets'])
            if not chosen:
                output['status'] = 'no_files_in_profile'
                output['reason'] = 'No suitable small files for starter mode; use --profile full with an adequate disk budget.'
                continue
            for asset in chosen:
                entry = {'filename': asset['filename'], 'bytes': asset['bytes'],
                         'estimated_bytes':asset.get('estimated_bytes'),
                         'publisher_checksum': f"{asset['checksum_algorithm']}:{asset['checksum']}" if asset['checksum'] else None}
                output['assets'].append(entry)
                size = asset['bytes'] or asset.get('estimated_bytes') or 0
                if size > max_file_gb * 1e9:
                    entry.update(status='skipped_file_budget', reason=f'Max file {max_file_gb:g} GB; increase --max-file-gb')
                    continue
                if aggregate + size > max_total_gb * 1e9:
                    entry.update(status='skipped_total_budget', reason=f'Max total {max_total_gb:g} GB; increase --max-total-gb')
                    continue
                if dry_run:
                    entry['status'] = 'planned'
                    aggregate += size
                    continue
                try:
                    allowance = int(min(max_file_gb * 1e9, max_total_gb * 1e9 - aggregate))
                    got = fetch_asset(asset, root / sid, max_download_bytes=allowance)
                    entry.update(got)
                    if got['status'] == 'downloaded':
                        aggregate += got['bytes']
                    if unpack and got.get('path') and Path(got['path']).suffix.lower() in ('.zip', '.xz', '.gz', '.tar'):
                        entry['extraction'] = safe_unpack(Path(got['path']), root / sid / 'unpacked' / Path(asset['filename']).stem)
                except Exception as exc:
                    entry.update(status='error', error=str(exc))
            states = [e['status'] for e in output['assets']]
            if all(s in ('downloaded', 'already_verified', 'already_exists_size_checked', 'planned') for s in states):
                output['status'] = 'planned' if dry_run else 'complete_selected'
            elif any(s in ('downloaded', 'already_verified', 'already_exists_size_checked', 'planned') for s in states):
                output['status'] = 'partial'
            else:
                output['status'] = 'no_files_downloaded'
        except Exception as exc:
            output.update(status='discovery_error', error=f'{type(exc).__name__}: {exc}')
    result['scheduled_bytes'] = aggregate
    if not dry_run:
        root.mkdir(parents=True, exist_ok=True)
        report = root / 'download_report.json'
        report.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
        result['report_path'] = str(report)
    return result
