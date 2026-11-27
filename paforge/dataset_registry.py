"""Audited, typed dataset roles.  Registration is not format verification.

The `lanes` and `ready` fields are *capabilities* of this application, not a
claim of clinical or publisher validation.  No clinical raw arrays are treated
as equivalent to material-class segmentations or independently measured optics.
"""
from __future__ import annotations

CATALOG = {
    'openqpai': {
        'title':'OpenQPAIData — multi-device tissue-mimicking phantoms',
        'summary':'InVision and TROPUS provide measured PA time-series and reconstructed/segmented phantom images; SVOT provides 3D reference volumes without shared raw signals. DIS provides independent material-property measurements.',
        'lanes':['phantom_material_learning','optical_profile','measured_signal_reconstruction','reference_visualization'],
        'excluded':['SVOT as measured detector time series','material labels as biological vessel classes','DIS arrays automatically merged without verified material mapping'],
        'access':'Zenodo 14044853; automatic verified-asset download',
        'adapter':'Multidevice InVision/TROPUS/SVOT/DIS pipeline',
        'license':'Consult publisher license, normally CC BY 4.0.',
        'status':'specialized pipeline',
    },
    'digital-twin-iqa':{
        'title':'Digital twins — photoacoustic reconstruction quality',
        'summary':'A QA reference dataset with IPASC acquisition examples and full-reference digital twins. Keep QA data outside the generative training pool.',
        'lanes':['reconstruction_quality_assessment','reference_visualization'],
        'excluded':['material-CVAE training by default','assuming digital twins are human measured acquisitions'],
        'access':'Zenodo 15388429; automatic download',
        'adapter':'IPASC inventory / normalized planar acquisition',
        'license':'Verify the release license.',
        'status':'download + IPASC inspection; quantitative QA schema not yet certified',
    },
    'oadat':{
        'title':'OADAT — forearm optoacoustic acquisition and simulation',
        'summary':'Experimental single-/multiwavelength forearm data and simulated data, with source-dependent raw PA signals. Hosted on ETH Libdrive; distribution may require user selection or authorized access.',
        'lanes':['measured_signal_reconstruction','synthetic_signal_benchmark','reference_visualization'],
        'excluded':['phantom material-label training','invented detector positions or timing'],
        'access':'Publisher-directed Libdrive access; optional local-file registration',
        'adapter':'Local ZIP/HDF5/MAT inventory + verified-key previews; full vendor-specific raw conversion needs a validated schema',
        'license':'Reported noncommercial; check repository terms.',
        'status':'guided download, role-aware local inspector; raw format not publisher-certified',
    },
    'paiskintone':{
        'title':'PAISKINTONE — multispectral in-vivo PA with varied skin tones',
        'summary':'Measured Acuity Echo CE observations from human participants across anatomical sites, with wavelengths 660–1300 nm. HDF5 archives inside per-subject TAR.XZ downloads; raw acquisitions are in-vivo, not phantom material maps.',
        'lanes':['measured_signal_visualization','spectral_analysis','measured_signal_reconstruction_if_calibrated'],
        'excluded':['material CVAE','learned phantom optics without measurement model','cross-subject data leakage'],
        'access':'Zenodo 20068219; per-subject archives and checksums',
        'adapter':'TAR.XZ/HDF5 inventory, on-demand single-member extraction, inspection and previews',
        'license':'Verify publisher record and usage terms for human research data.',
        'status':'download + nested archive inspection; Acuity raw reconstruction requires schema review',
    },
    'synthetic-skin':{
        'title':'Synthetic Vascularized Digital Skin Phantoms',
        'summary':'60,000 synthetic 2D digital skin samples in 20 HDF5 shards with aligned absorption (mua), fluence, initial pressure (IP), and simulated acoustic sinograms; 100-sample testing shard available.',
        'lanes':['optical_initial_pressure_training','simulated_signal_benchmark','reference_visualization'],
        'excluded':['claiming experimental signals','converting continuous mua arrays to categorical tissue labels automatically'],
        'access':'Zenodo 21785486; sample subset or individual 5-GB shards',
        'adapter':'Lazy HDF5 paired optical/signal inspection with output schema verification',
        'license':'Check publisher license and MATLAB simulation code terms.',
        'status':'paired arrays preview + provenance; full optical CVAE integration pending schema validation',
    },
    'breast-pact':{
        'title':'Breast PACT/URCT — clinical paired image dataset',
        'summary':'Dryad archive containing 161 paired human breast tumor PACT and URCT image sets (322 MAT files). No verified detector time-series metadata in the published description.',
        'lanes':['clinical_reconstruction_reference','image_domain_quality_assessment'],
        'excluded':['raw signal reconstruction absent verified raw data','material-label learning','treating ultrasound CT as optical measurements'],
        'access':'Dryad DOI 10.5061/dryad.vmcvdnd55',
        'adapter':'ZIP/MAT image inventory and visualization for selected numeric arrays',
        'license':'Verify the Dryad usage and privacy terms.',
        'status':'download + image-domain inspector; raw data not asserted',
    },
    'cambridge-vascular':{
        'title':'Cambridge vascular-network mesoscopy — Brown et al.',
        'summary':'Synthetic L-system vascular masks, optical/acoustic reconstructions, experimental string-phantom/RSOM images and vessel segmentation outputs; the RawRSOM label denotes reconstructed images, not raw detector waveforms.',
        'lanes':['vascular_segmentation_reference','vessel_morphology_learning','image_quality_assessment'],
        'excluded':['detector-time-series reconstruction from RawRSOM images','optical-property material CVAE'],
        'access':'Cambridge DOI 10.17863/CAM.78208; individual bitstreams',
        'adapter':'ZIP/HDF5 mask/image inspector, role-separated from raw signals',
        'license':'CC BY-NC 4.0 (noncommercial).',
        'status':'individual publisher bitstream download + image inspector',
    },
    'cambridge-patato':{
        'title':'Cambridge PATATO example acquisitions — Else et al.',
        'summary':'Experimental HDF5 Acuity/inVision phantom data and in-vivo mouse oxygen-challenge / contrast-enhanced acquisitions; data include raw_data, recons and GEOMETRY/wavelength metadata.',
        'lanes':['measured_signal_visualization','measured_signal_reconstruction_if_calibrated','reference_visualization'],
        'excluded':['treating mouse in-vivo data as material phantom maps','invented geometry, timing or wavelength indices'],
        'access':'Cambridge DOI 10.17863/CAM.93181; DSpace metadata-based individual download',
        'adapter':'PATATO HDF5 schema inspector, sample visualization and explicit geometry-gated normalization',
        'license':'CC BY 4.0 according to Cambridge repository.',
        'status':'dynamic Cambridge API download + PATATO-aware inspection',
    },
    'shanghaitech-vessel':{
        'title':'ShanghaiTech — vessel phantom photoacoustic tomography',
        'summary':'7.7-GB Vessel_Data.zip research deposit. Its detailed array layout, sampling frequency and detector geometry are not established from the landing page alone.',
        'lanes':['vessel_phantom_visualization','reconstruction_if_raw_metadata_verified'],
        'excluded':['assuming unverified arrays are sensor time-series','automatic material-optics training'],
        'access':'Zenodo 10719987; publisher MD5 checksum',
        'adapter':'ZIP and array inventory; block raw reconstruction until signal axes and geometry verified',
        'license':'Verify license on Zenodo record.',
        'status':'download + schema-driven inspection (uncertified raw adapter)',
    },
    'pam3':{
        'title':'PAM3 photoacoustic/ultrasound 3D tomography',
        'summary':'Reconstructed 3D optical-acoustic/ultrasound image data and MATLAB visualization scripts. Raw signals not confirmed.',
        'lanes':['volumetric_reference_visualization'],
        'excluded':['unverified raw time-series','2D acoustic simulation from 3D images without defined projection'],
        'access':'Zenodo 20612980',
        'adapter':'Numeric image/volume array inspection',
        'license':'Verify publisher asset terms.',
        'status':'download + image inspector',
    },
    'photiqa':{
        'title':'PhotIQA — photoacoustic image quality assessment',
        'summary':'Quality-rating reference images and annotations. Intended for independent image quality analysis, not generative optical training.',
        'lanes':['image_quality_assessment'],
        'excluded':['material map estimation','invented sensor signals'],
        'access':'Zenodo 15828190',
        'adapter':'Image/annotation inspection',
        'license':'Check upstream license.',
        'status':'download + image inspector',
    },
    'zenodo-23348355':{
        'title':'User-supplied Zenodo record 23348355 (contents unverified)',
        'summary':'The user supplied this record URL. Its identity, availability, file listing and MATLAB/Python contents could not be independently authenticated in the present environment.',
        'lanes':['publisher_verified_file_download','unclassified_local_file_inspection'],
        'excluded':['automatic PATbox attribution','executing untrusted MATLAB/Python code','assuming experimental acquisitions'],
        'access':'Zenodo 23348355; runtime publisher API lookup only',
        'adapter':'Zenodo checksum-aware download; content classified only after successful metadata discovery',
        'license':'Not verified; user must review record.',
        'status':'unverified external source; no claim of complete algorithm integration',
    },
}


def describe(source_id: str) -> dict:
    if source_id not in CATALOG:
        raise KeyError('Unknown source: ' + source_id)
    return {'id':source_id,**CATALOG[source_id]}


def all_descriptions() -> list[dict]:
    return [describe(k) for k in CATALOG]
