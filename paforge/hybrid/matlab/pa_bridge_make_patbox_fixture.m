function fixtureFile = pa_bridge_make_patbox_fixture(simFile, exchangeFile, verifiedPositions_m, fixtureFile, imageGrid)
%PA_BRIDGE_MAKE_PATBOX_FIXTURE Bind existing PATBox sim to a canonical export.
% Strict reference path: the original sim.sensor_data MUST be identical to the
% exported HDF5 acquisition. No silent signal substitution or channel mapping.
% imageGrid: optional reviewed struct {dx_m,dy_m,origin_x_m,origin_y_m,orientation}.
arguments
    simFile (1,1) string
    exchangeFile (1,1) string
    verifiedPositions_m (:,3) double
    fixtureFile (1,1) string
    imageGrid (1,1) struct = struct()
end
assert(~isfile(fixtureFile), 'PAHybrid:Exists', 'Refusing to overwrite fixture');
s = load(simFile,'sim');
assert(isfield(s,'sim'),'PAHybrid:Sim','Missing sim structure');
sim = s.sim;
a = pahybrid.read_acquisition(exchangeFile);
assert(isfield(sim,'sensor_data') && isfield(sim,'sensor') && ...
    isfield(sim,'kgrid') && isfield(sim,'sound_speed'), ...
    'PAHybrid:Sim', 'Missing core PATBox simulation fields');
originalTraces = pa_pr04_extract_traces(sim);
assert(isequal(size(originalTraces),size(a.signals)) && ...
    isequal(originalTraces,double(a.signals)), ...
    'PAHybrid:Source', 'Original PATBox signal samples differ from HDF5; no reference possible');
assert(isequal(size(verifiedPositions_m),size(a.detectors_m)) && ...
    max(abs(verifiedPositions_m(:)-a.detectors_m(:)))<=1e-12, ...
    'PAHybrid:Geometry', 'Verified detector order differs from acquisition');
assert(isprop(sim.kgrid,'dt') && ...
    abs(double(sim.kgrid.dt)-a.sample_interval_s)<=max(1e-14, 1e-6*a.sample_interval_s), ...
    'PAHybrid:Clock','Sample interval differs');
if ~isempty(fieldnames(imageGrid))
    fields = {'dx_m','dy_m','origin_x_m','origin_y_m','orientation'};
    assert(all(isfield(imageGrid, fields)), 'PAHybrid:ImageGrid', ...
        'All 5 image orientation/grid fields required');
end
sim.bridge_acquisition_id = char(string(a.metadata.acquisition_id));
sim.bridge_detector_positions_m = a.detectors_m;
sim.bridge_coordinate_frame = char(string(a.metadata.coordinate_frame));
sim.bridge_validation_mode = 'reference';
sim.bridge_image_grid = imageGrid;
save(fixtureFile,'sim','-v7.3');
fprintf('Verified PATBox sim fixture saved: %s\n', fixtureFile);
end
