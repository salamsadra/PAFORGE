function pa_pr04_preflight()
% MATLAB runtime smoke: checks available API without claiming full parity.
assert(exist('patSimulate','file') == 2, 'PAForge:PATBox', 'patSimulate missing');
assert(exist('patReconstruct','file') == 2, 'PAForge:PATBox', 'patReconstruct missing');
assert(exist('kspaceFirstOrder2D','file') == 2, 'PAForge:kWave', 'k-Wave missing');
fprintf('PR04 MATLAB preflight OK: %s | PATBox %s\n',version,which('patReconstruct'));
end
