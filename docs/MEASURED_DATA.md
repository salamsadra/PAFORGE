# Measured photoacoustic data

Use compatible detector time series and acquisition metadata. Do not treat reconstructed images as raw signals.

After obtaining OpenQPAIData InVision separately and reviewing its provenance and license, choose an existing phantom and wavelength:

```bash
paforge extract-invision /path/to/invision.zip --phantom P.5.10.2 \
  --wavelength-nm 700 --output data/selected_invision --attest-experimental

paforge reconstruct-acquisition data/selected_invision/native/P.5.10.2_700.h5 \
  --output outputs/measured --method das --fov-m 0.032 --grid-size 96 \
  --sound-speed-m-s 1489 --time-decimate 1
```

The reconstruction uses imported detector positions and sampling metadata. The speed of sound and field of view are illustrative settings, not scanner calibration. Publisher reference images and labels remain separate. For other planar IPASC files, see `paforge import-ipasc --help`. For general HDF5 arrays, see `paforge extract-acquisition --help` and provide the true axis layout, sampling rate and detector coordinates. Reconstruction is qualitative unless independently validated.
