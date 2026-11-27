function outputFile = pa_bridge_roundtrip(inputFile, outputFile)
%PA_BRIDGE_ROUNDTRIP MATLAB HDF5 byte-value roundtrip, NOT physical validation.
% MATLAB must independently be run by the researcher. Does not verify SHA256;
% Python read_acquisition() verifies SHA256 on the returned file.
arguments
    inputFile (1,1) string
    outputFile (1,1) string
end
assert(isfile(inputFile), 'PAHybrid:Input', 'Input HDF5 file missing');
assert(~isfile(outputFile), 'PAHybrid:Exists', 'Refusing to overwrite output HDF5');
a = pahybrid.read_acquisition(inputFile);
N = a.n_detectors; T = a.n_samples;
% HDF5 vectors, no ambiguous 2-D dimension permutation.
sig = reshape(a.signals.', [], 1);
pos = reshape(a.detectors_m.', [], 1);
h5create(outputFile, '/acquisition/signals_flat_c', [N*T, 1], 'Datatype', 'double');
h5write(outputFile, '/acquisition/signals_flat_c', sig);
h5create(outputFile, '/acquisition/detectors_flat_c', [N*3, 1], 'Datatype', 'double');
h5write(outputFile, '/acquisition/detectors_flat_c', pos);
attrs = {'n_detectors','n_samples','signal_axis_order','detector_axis_order', ...
    'sample_interval_s','detector_coordinate_units','signal_units', ...
    'signal_sha256_c_f64','detector_sha256_c_f64','metadata_json'};
for k = 1:numel(attrs)
    key = attrs{k};
    value = h5readatt(inputFile,'/acquisition',key);
    h5writeatt(outputFile,'/acquisition',key,value);
end
h5writeatt(outputFile,'/','format',char(h5readatt(inputFile,'/','format')));
h5writeatt(outputFile,'/','schema_version',char(h5readatt(inputFile,'/','schema_version')));
h5writeatt(outputFile,'/','producer','MATLAB:pa_bridge_roundtrip');
% The SHA256 values are copied, not recomputed. Python verifies unchanged bytes.
fprintf('PR03 MATLAB roundtrip written: %s\n', outputFile);
end
