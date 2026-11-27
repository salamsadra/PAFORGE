function simFile = pa_pr04_generate_simulation(patboxRoot, simFile)
%PA_PR04_GENERATE_SIMULATION Execute actual PATBox/k-Wave; requires MATLAB.
% Not runnable in the Python-only CI environment. Generates *simulated* data.
arguments
    patboxRoot (1,1) string
    simFile (1,1) string
end
assert(isfolder(patboxRoot), 'PAForge:PATBox', 'PATBox folder does not exist');
addpath(genpath(char(patboxRoot)));
assert(exist('patSimulate','file') == 2 && ...
       exist('kspaceFirstOrder2D','file') == 2, 'PAForge:Dependencies', ...
       'Add original PATBox and installed k-Wave to the MATLAB path');
assert(~isfile(simFile),'PAForge:Exists','Refusing to overwrite simulation');
imageFile = fullfile(patboxRoot,'data','Example1.bmp');
assert(isfile(imageFile), 'PAForge:Data', 'PATBox example image missing');
% Keep a low-memory deterministic reference where possible. PATBox may also
% rely on its params.yaml; preserve a copy and record PATBox/k-Wave commits.
rng(20261116,'twister'); % local MATLAB RNG only; archive environment and seeds
sim=patSimulate(imageFile,'SensorType','linear','GridSize',128, ...
    'NumTransducers',32,'TargetSNRdB',30);
assert(isfield(sim,'sensor_data') && isfield(sim,'kgrid'), ...
    'PAForge:Simulation','PATBox simulation did not produce expected fields');
save(simFile,'sim','-v7.3');
fprintf('Original PATBox simulation saved: %s\n',simFile);
end
