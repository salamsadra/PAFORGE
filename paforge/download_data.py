"""Explicit, checksum-verified public experimental archive download helpers.

Never auto-download data at import or install time. OpenQPAIData source data are
licensed by upstream authors; preserve the upstream README and credit them.
"""
import hashlib
import shutil
import urllib.request
import zipfile
from datetime import datetime, timezone
from uuid import uuid4
from pathlib import Path

OPENQPAI = {
    'invision':('invision.zip','a3ab81d0256feb2ffa8412b3ed40efa6'),
    'tropus':('tropus.zip','e826db69f96eeeadb7e18a459114b9e9'),
    'dis':('DIS.zip','3ad5e85f91aed6fbaad978d8a8d94515'),
}
BASE='https://zenodo.org/records/14044853/files/'


def _md5_file(path):
    hasher=hashlib.md5()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda:source.read(4*1024*1024),b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def _quarantine(path):
    """Preserve an untrusted file without overwriting another file."""
    path=Path(path)
    suffix=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    quarantine=path.with_name(f'{path.name}.invalid-{suffix}-{uuid4().hex[:8]}')
    path.replace(quarantine)
    return quarantine


def fetch_openqpai(target, kind='invision', unpack=False):
    if kind not in OPENQPAI:raise ValueError(f"Choose one of {sorted(OPENQPAI)}")
    filename,md5=OPENQPAI[kind]
    target=Path(target);target.mkdir(parents=True,exist_ok=True)
    dest=target/filename
    if dest.exists():
        current_md5=_md5_file(dest)
        if current_md5!=md5:
            _quarantine(dest)
            # A future invocation can fetch a clean copy. This invocation
            # deliberately stops so that corruption is never hidden.
            raise ValueError(f'Cached archive checksum mismatch: expected {md5}, found {current_md5}; invalid cache quarantined. Retry download.')
    else:
        temp=target/(filename+'.'+uuid4().hex+'.partial')
        req=urllib.request.Request(BASE+filename+'?download=1',headers={'User-Agent':'PA-FORGE-research/0.3'})
        try:
            with urllib.request.urlopen(req,timeout=40) as response, temp.open('wb') as out:
                shutil.copyfileobj(response,out,1024*1024)
            fetched_md5=_md5_file(temp)
            if fetched_md5!=md5:
                quarantine=_quarantine(temp)
                raise ValueError(f'Download checksum mismatch: expected {md5}, found {fetched_md5}; untrusted bytes saved as {quarantine.name}, no final archive created.')
            # Only authenticated bytes acquire the published final name.
            temp.replace(dest)
        except Exception:
            temp.unlink(missing_ok=True)
            raise
    unpack_dir=None
    if unpack:
        unpack_dir=target/kind
        unpack_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(dest) as z:
            for member in z.infolist():
                path=(unpack_dir/member.filename).resolve()
                if not path.is_relative_to(unpack_dir.resolve()):
                    raise ValueError('Unsafe archive path detected')
                # Reject Unix symbolic links in zip archives
                if (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError('Symlink in zip not permitted')
            z.extractall(unpack_dir)
    return {'archive':str(dest),'upstream_md5_verified':True,'unpacked_to':str(unpack_dir) if unpack else None,
            'source':'https://zenodo.org/records/14044853','license':'CC-BY-4.0 (upstream)' }
