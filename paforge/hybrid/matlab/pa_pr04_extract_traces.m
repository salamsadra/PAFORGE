function traces = pa_pr04_extract_traces(sim)
%PA_PR04_EXTRACT_TRACES Read supported PATBox signal payloads without changing SIM.
% Supports legacy numeric [channel,time] and modern struct sensor_data.p.
% Do not replace sim.sensor_data in the parent caller; PATBox owns its schema.
assert(isstruct(sim) && isscalar(sim) && isfield(sim, 'sensor_data'), ...
    'PAForge:Sim', 'PATBox sim.sensor_data is missing');
value = sim.sensor_data;
if isstruct(value)
    assert(isscalar(value) && isfield(value,'p'), ...
        'PAForge:SignalFormat','Expected scalar sensor_data struct with field p');
    value = value.p;
end
assert(isnumeric(value) && ismatrix(value) && isreal(value) && ...
    size(value,1) >= 2 && size(value,2) >= 16 && ...
    all(isfinite(value(:))), 'PAForge:Signals', ...
    'Expected finite real numeric [channel,time] traces, or sensor_data.p');
traces = double(value);
end
