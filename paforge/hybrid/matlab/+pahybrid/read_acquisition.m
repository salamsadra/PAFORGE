function acq = read_acquisition(filePath)
%READ_ACQUISITION Read the PA-HYBRID flattened HDF5 exchange format.
% Flattened C-order data avoids dimension reversal between h5py and MATLAB.
% THIS IS A CROSS-LANGUAGE CONTRACT, not an IPASC-conformant reader.
arguments
    filePath (1,1) string
end
fmt = string(h5readatt(filePath, '/', 'format'));
ver = string(h5readatt(filePath, '/', 'schema_version'));
assert(fmt == "PA-HYBRID-EXCHANGE" && ver == "0.1.0", ...
    'PAHybrid:Schema', 'Unsupported HDF5 schema/version');
base = '/acquisition';
nd = double(h5readatt(filePath, base, 'n_detectors'));
nt = double(h5readatt(filePath, base, 'n_samples'));
assert(nd >= 2 && nt >= 16 && nd == floor(nd) && nt == floor(nt), ...
    'PAHybrid:Shape', 'Invalid detector/time counts');
assert(string(h5readatt(filePath, base, 'signal_axis_order')) == "detector,time", ...
    'PAHybrid:Axis', 'Unexpected signal axis contract');
assert(string(h5readatt(filePath, base, 'detector_axis_order')) == "detector,xyz", ...
    'PAHybrid:Axis', 'Unexpected detector coordinate contract');
assert(string(h5readatt(filePath, base, 'detector_coordinate_units')) == "m", ...
    'PAHybrid:Units', 'Sensor coordinates must be metres');
sig = double(h5read(filePath, base + "/signals_flat_c"));
pos = double(h5read(filePath, base + "/detectors_flat_c"));
assert(numel(sig) == nd*nt && numel(pos) == nd*3, ...
    'PAHybrid:Size', 'Dataset lengths are inconsistent with metadata');
% Python to MATLAB: C-order flatten [nd,nt] -> transpose(reshape(flat,[nt,nd])).
acq.signals = reshape(sig(:), [nt, nd]).';
acq.detectors_m = reshape(pos(:), [3, nd]).';
acq.sample_interval_s = double(h5readatt(filePath, base, 'sample_interval_s'));
assert(isfinite(acq.sample_interval_s) && acq.sample_interval_s > 0, ...
    'PAHybrid:Clock', 'Invalid sample interval');
assert(all(isfinite(acq.signals(:))) && all(isfinite(acq.detectors_m(:))), ...
    'PAHybrid:Numeric', 'Signals or coordinates are not finite');
metadataJSON = char(h5readatt(filePath, base, 'metadata_json'));
acq.metadata = jsondecode(metadataJSON);
acq.metadata_json = metadataJSON;
acq.n_detectors = nd;
acq.n_samples = nt;
acq.signal_sha256_c_f64 = string(h5readatt(filePath, base, 'signal_sha256_c_f64'));
acq.detector_sha256_c_f64 = string(h5readatt(filePath, base, 'detector_sha256_c_f64'));
% Note: SHA256 re-verification occurs in Python; this MATLAB reader checks
% structural consistency but does not independently recompute a byte hash.
end
