function out = pa_pr04_reconstruct(simFile, exchangeFile, channelCsv, outDir, algorithm)
%PA_PR04_RECONSTRUCT Two distinct routes to the SAME PATBox algorithm.
% DIRECT calls the original PATBox sim; BRIDGE calls the HDF5-gated adapter.
% Both images must originate in MATLAB. This tests INTEGRATION PARITY ONLY.
arguments
    simFile (1,1) string
    exchangeFile (1,1) string
    channelCsv (1,1) string
    outDir (1,1) string
    algorithm (1,1) string = "DAS"
end
assert(isfile(simFile) && isfile(exchangeFile) && isfile(channelCsv), ...
    'PAForge:Inputs', 'Missing original PATBox sim, imported exchange, or CSV');
assert(exist('patReconstruct','file') == 2, 'PAForge:PATBox', ...
    'Original PATBox is not on the MATLAB path');
assert(any(strcmpi(algorithm, {'DAS','DMAS','UBP','FBP'})), 'PAForge:Algorithm', ...
    'PR-04 limits algorithms to DAS, DMAS, UBP and FBP');
S = load(simFile,'sim');
assert(isfield(S,'sim'), 'PAForge:Sim', 'MAT variable sim not found');
sim = S.sim;
a = pahybrid.read_acquisition(exchangeFile);
coords = readmatrix(channelCsv);
assert(size(coords,2) == 4 && isequal(coords(:,1), (1:size(coords,1)).') && ...
    size(coords,1) == a.n_detectors && all(isfinite(coords(:))), ...
    'PAForge:CSV', 'Channel CSV must exactly enumerate detectors');
originalTraces = pa_pr04_extract_traces(sim);
assert(isequal(size(originalTraces),size(a.signals)) && ...
    isequal(originalTraces,a.signals), 'PAForge:Waveforms', ...
    'Original PATBox and Python-imported traces differ');
assert(max(abs(coords(:,2:4)-a.detectors_m),[], 'all')<=1e-12, ...
    'PAForge:Geometry', 'Coordinates in CSV differ from HDF5');
assert(isprop(sim.kgrid,'dt') && ...
    abs(double(sim.kgrid.dt)-a.sample_interval_s)<=max(1e-14,1e-6*a.sample_interval_s), ...
    'PAForge:Time', 'PATBox and HDF5 clock mismatch');
assert(isprop(sim.kgrid,'dx') && isprop(sim.kgrid,'dy') && ...
    isprop(sim.kgrid,'Nx') && isprop(sim.kgrid,'Ny') && ...
    isprop(sim.kgrid,'x_vec') && isprop(sim.kgrid,'y_vec'), ...
    'PAForge:Grid', 'Need a k-Wave kgrid with explicit spacing and axis vectors');
if ~isfolder(outDir); mkdir(outDir); end
refFile = fullfile(outDir,'direct_patbox.h5');
bridgeFile = fullfile(outDir,'bridge_patbox.h5');
fixtureFile = fullfile(outDir,'verified_sim_fixture.mat');
assert(~isfile(refFile) && ~isfile(bridgeFile) && ~isfile(fixtureFile), ...
    'PAForge:Exists', 'No overwriting allowed in benchmark runs');
% This orientation is only valid if PATBox returns a native [Nx Ny] image.
% We test size BEFORE recording grid metadata; no silent resampling.
grid = struct('dx_m',double(sim.kgrid.dx),'dy_m',double(sim.kgrid.dy), ...
   'origin_x_m',double(sim.kgrid.x_vec(1)), ...
   'origin_y_m',double(sim.kgrid.y_vec(1)), ...
   'orientation','rows=x,cols=y (native k-Wave)');
% Direct reconstruction without invoking the PAForge adapter:
[p0,~] = patReconstruct(sim,'Algorithm',char(algorithm),'SaveOutput',false);
assert(isequal(size(p0),[double(sim.kgrid.Nx),double(sim.kgrid.Ny)]), ...
    'PAForge:Grid', 'Output does not match native [Nx Ny] k-Wave image grid');
pahybrid.write_result(refFile,p0,algorithm,string(a.metadata.acquisition_id),sim.kgrid,grid);
h5writeatt(refFile,'/','producer','MATLAB:PATBox:direct');
h5writeatt(refFile,'/','acquisition_signal_sha256',char(a.signal_sha256_c_f64));
h5writeatt(refFile,'/','matlab_release',version('-release'));
h5writeatt(refFile,'/','patbox_entrypoint',which('patReconstruct'));
% Bridge: input to PATBox must now survive HDF5, MATLAB reader and fixture checks.
pa_bridge_make_patbox_fixture(simFile,exchangeFile,coords(:,2:4),fixtureFile,grid);
pa_bridge_run_patbox(exchangeFile,fixtureFile,algorithm,bridgeFile);
h5writeatt(bridgeFile,'/','producer','MATLAB:PAForge:bridge');
h5writeatt(bridgeFile,'/','acquisition_signal_sha256',char(a.signal_sha256_c_f64));
h5writeatt(bridgeFile,'/','matlab_release',version('-release'));
h5writeatt(bridgeFile,'/','patbox_entrypoint',which('patReconstruct'));
% Machine-readable execution provenance, never call it a physical validation.
out = struct('direct',refFile,'bridge',bridgeFile,'fixture',fixtureFile, ...
    'algorithm',algorithm,'matlab_version',version,'patbox_path',which('patReconstruct'));
fileID=fopen(fullfile(outDir,'matlab_environment.txt'),'w');
assert(fileID>0,'PAForge:Filesystem','Could not write environment report');
cleanup=onCleanup(@()fclose(fileID));
fprintf(fileID,'MATLAB: %s\nRelease: %s\nPATBox: %s\nk-Wave: %s\n', ...
    version,version('-release'),which('patReconstruct'),which('kspaceFirstOrder2D'));
end
