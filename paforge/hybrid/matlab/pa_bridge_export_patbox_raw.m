function outputFile = pa_bridge_export_patbox_raw(sim, detectorPositions_m, outputFile, acquisitionID, deviceID, sessionID, coordinateFrame, verificationNote)
%PA_BRIDGE_EXPORT_PATBOX_RAW Export raw PATBox arrays without inventing hashes.
% Prerequisite: independently verify detectorPositions_m row order corresponds
% exactly to sim.sensor_data rows, and the frame/units match the real sensor.
% This is an intermediate representation, not PA-HYBRID-EXCHANGE or IPASC.
arguments
    sim (1,1) struct
    detectorPositions_m (:,3) double
    outputFile (1,1) string
    acquisitionID (1,1) string
    deviceID (1,1) string
    sessionID (1,1) string
    coordinateFrame (1,1) string
    verificationNote (1,1) string
end
assert(strlength(verificationNote) >= 15, 'PAHybrid:Verification', ...
    'Provide a real detector/channel-order verification note (>=15 characters)');
assert(isfield(sim,'sensor_data') && isfield(sim,'kgrid'), 'PAHybrid:Sim', ...
    'PATBox sim must have sensor_data and kgrid');
traces = pa_pr04_extract_traces(sim);
assert(ismatrix(traces) && isreal(traces) && all(isfinite(traces(:))), ...
    'PAHybrid:Signals', 'Expected finite real traces');
[N,T] = size(traces);
assert(N >= 2 && T >= 16 && size(detectorPositions_m,1) == N && ...
    all(isfinite(detectorPositions_m(:))), 'PAHybrid:Shape', 'Detector order/count invalid');
assert(isprop(sim.kgrid,'dt') && isfinite(sim.kgrid.dt) && sim.kgrid.dt>0, ...
    'PAHybrid:Clock','Expected k-Wave kgrid with positive dt');
assert(~isfile(outputFile), 'PAHybrid:Exists','Refusing to overwrite raw export');
sigflat = reshape(traces.',[],1);
posflat = reshape(detectorPositions_m.',[],1);
h5create(outputFile,'/raw/signals_flat_c',[N*T 1],'Datatype','double');
h5write(outputFile,'/raw/signals_flat_c',sigflat);
h5create(outputFile,'/raw/detectors_flat_c',[N*3 1],'Datatype','double');
h5write(outputFile,'/raw/detectors_flat_c',posflat);
h5writeatt(outputFile,'/','format','PA-PATBOX-RAW-EXPORT');
h5writeatt(outputFile,'/','schema_version','0.1.0');
h5writeatt(outputFile,'/raw','n_detectors',int64(N));
h5writeatt(outputFile,'/raw','n_samples',int64(T));
h5writeatt(outputFile,'/raw','sample_interval_s',double(sim.kgrid.dt));
h5writeatt(outputFile,'/raw','acquisition_id',char(acquisitionID));
h5writeatt(outputFile,'/raw','device_id',char(deviceID));
h5writeatt(outputFile,'/raw','session_id',char(sessionID));
h5writeatt(outputFile,'/raw','coordinate_frame',char(coordinateFrame));
h5writeatt(outputFile,'/raw','channel_mapping_note',char(verificationNote));
h5writeatt(outputFile,'/raw','signal_units','unspecified');
fprintf('PATBox raw exported; import to Python to attach SHA256: %s\n',outputFile);
end
