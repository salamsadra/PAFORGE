function pa_pr04_struct_payload_test()
% No k-Wave needed: tests matrix vs. struct sensor data interchange only.
a = single(reshape(1:96,[3,32]));
s1=struct('sensor_data',a);
s2=struct('sensor_data',struct('p',a));
assert(isequal(pa_pr04_extract_traces(s1),double(a)));
assert(isequal(pa_pr04_extract_traces(s2),double(a)));
assert(isequal(size(pa_pr04_extract_traces(s2)),[3 32]));
try
    pa_pr04_extract_traces(struct('sensor_data',struct('q',a)));
    error('PAForge:Test','Expected rejection of missing p');
catch ME
    assert(strcmp(ME.identifier,'PAForge:SignalFormat'));
end
fprintf('PR04 payload compatibility smoke PASS (matrix and struct.p)\n');
end
