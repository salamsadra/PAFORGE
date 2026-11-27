function rawFile = pa_pr04_prepare(simFile, channelCsv, outDir, acquisitionID, verificationNote)
%PA_PR04_PREPARE Export real PATBox-generated traces for Python ingestion.
% simFile must contain an ORIGINAL PATBox sim saved with the variable name sim.
% channelCsv must be audited row-by-row: column order channel,x_m,y_m,z_m.
% This stage is NOT a MATLAB/Python parity result or physics validation.
arguments
    simFile (1,1) string
    channelCsv (1,1) string
    outDir (1,1) string
    acquisitionID (1,1) string
    verificationNote (1,1) string
end
assert(isfile(simFile) && isfile(channelCsv), 'PAForge:Input', ...
    'Original PATBox sim and audited channel-index CSV are required');
assert(strlength(verificationNote) >= 30, 'PAForge:Review', ...
    'Document how channel order and physical coordinates were independently checked');
s = load(simFile, 'sim');
assert(isfield(s, 'sim') && isfield(s.sim, 'sensor_data'), 'PAForge:Sim', ...
    'The .mat file must contain a PATBox sim with sensor_data');
coords = readmatrix(channelCsv);
assert(size(coords,2) == 4 && all(isfinite(coords(:))), 'PAForge:CSV', ...
    'CSV must contain channel_index,x_m,y_m,z_m as four finite columns');
traces = pa_pr04_extract_traces(s.sim);
N = size(traces,1);
assert(size(coords,1) == N && isequal(coords(:,1), (1:N).'), ...
    'PAForge:ChannelOrder', 'CSV channels must be exactly 1...N in raw-signal row order');
if ~isfolder(outDir); mkdir(outDir); end
rawFile = fullfile(outDir,'patbox_raw.h5');
pa_bridge_export_patbox_raw(s.sim,coords(:,2:4),rawFile, ...
    acquisitionID,"PATBox-simulation","PR04-reference","PATBox-audited-xyz-m", verificationNote);
end
