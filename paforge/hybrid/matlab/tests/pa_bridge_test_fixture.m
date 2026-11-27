function pa_bridge_test_fixture(filePath)
% MATLAB-only acceptance test for the Python-created deterministic fixture.
acq = pahybrid.read_acquisition(string(filePath));
assert(all(size(acq.signals) == [4,64]));
assert(all(size(acq.detectors_m) == [4,3]));
assert(abs(acq.signals(1,1) - 0.0) < 1e-12);
assert(abs(acq.signals(1,2) - 0.1) < 1e-12);
assert(abs(acq.signals(2,1) - 6.4) < 1e-12);
assert(abs(acq.detectors_m(4,1) - 0.0012) < 1e-12);
assert(pa_bridge_validate(string(filePath)));
fprintf('PA-HYBRID MATLAB fixture: PASS\n');
end
