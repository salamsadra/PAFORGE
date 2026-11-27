function report = pa_pr04_audit_geometry(simFile, outDir)
%PA_PR04_AUDIT_GEOMETRY Inspect raw channel geometry and create a CANDIDATE CSV.
% DOES NOT independently prove that PATBox sensor_data.p row order equals the
% order of sensor_geometry_actual.positions. It never produces channels.csv.
arguments
    simFile (1,1) string
    outDir (1,1) string
end
assert(isfile(simFile),'PAForge:SimFile','PATBox sim .mat not found');
S = load(simFile,'sim');
assert(isfield(S,'sim'),'PAForge:Sim','Missing sim in MAT file');
sim = S.sim;
raw = pa_pr04_extract_traces(sim);
N = size(raw,1);
assert(isfield(sim,'sensor_geometry_actual') && ...
    isstruct(sim.sensor_geometry_actual) && ...
    isfield(sim.sensor_geometry_actual,'positions'), ...
    'PAForge:Geometry','Missing measured/actual geometry positions');
XY = double(sim.sensor_geometry_actual.positions);
assert(isequal(size(XY),[N 2]) && all(isfinite(XY(:))), ...
    'PAForge:Geometry','Actual positions are not finite [N,2]');
fprintf('\nPR04 sensor data: %d channels x %d time samples\n',N,size(raw,2));
fprintf('Actual positions (assumed model units; VERIFY SI metres):\n');
fprintf('   x range: %.9g .. %.9g\n',min(XY(:,1)),max(XY(:,1)));
fprintf('   y range: %.9g .. %.9g\n',min(XY(:,2)),max(XY(:,2)));
assert(isfield(sim,'sensor') && isstruct(sim.sensor), ...
    'PAForge:Sensor','sim.sensor missing');
candidates = {'element_positions','physical_element_positions', ...
              'nominal_element_positions'};
for j=1:numel(candidates)
    key=candidates{j};
    if isfield(sim.sensor,key)
        B=double(sim.sensor.(key));
        if isequal(size(B),size(XY)) && all(isfinite(B(:)))
            fprintf('max abs(actual.positions - sensor.%s): %.12g\n', ...
                key,max(abs(XY(:)-B(:))));
        else
            fprintf('sensor.%s: shape incompatible or non-finite\n',key);
        end
    end
end
if isfield(sim.sensor,'mask')
    mask=sim.sensor.mask;
    fprintf('sensor.mask: %s, nonzero=%d, size=[%d %d]\n', ...
        class(mask),nnz(mask),size(mask,1),size(mask,2));
end
if isfield(sim.sensor_geometry_actual,'combine_mode')
    fprintf('combine_mode: %s\n',char(string(sim.sensor_geometry_actual.combine_mode)));
end
if isfield(sim.sensor_geometry_actual,'num_simulation_points')
    fprintf('simulation points: %g\n', ...
        double(sim.sensor_geometry_actual.num_simulation_points));
end
if isfield(sim,'kgrid') && isprop(sim.kgrid,'dt')
    fprintf('dt = %.12g seconds\n',double(sim.kgrid.dt));
end
if ~isfolder(outDir);mkdir(outDir);end
candidate = fullfile(outDir,'channels_CANDIDATE.csv');
assert(~isfile(candidate),'PAForge:Exists', ...
    'Candidate CSV exists; review it before creating a new file');
coords=[(1:N).' XY zeros(N,1)];
writematrix(coords,candidate);   % NO HEADER: [channel x_m y_m z_m]
fprintf('\nCANDIDATE ONLY (not audited): %s\n',candidate);
fprintf(['Do NOT rename as channels.csv until you have confirmed that\n' ...
         'a) the coordinates are in metres and the required physical frame;\n' ...
         'b) each geometry row corresponds to sensor_data.p trace row.\n']);
report=struct('candidate',candidate,'n_channels',N,'n_samples',size(raw,2), ...
              'range_x', [min(XY(:,1)) max(XY(:,1))], ...
              'range_y',[min(XY(:,2)) max(XY(:,2))], ...
              'review_status','PENDING_CHANNEL_MAPPING_REVIEW');
end
