function pa_bridge_pr03_smoke(inputFile, outputFile)
% MATLAB suite for independently generated Python non-square fixture.
% Example:
% addpath('/path/to/installed/paforge/hybrid/matlab');
% pa_bridge_pr03_smoke("fixture.h5","returned.h5")
arguments
    inputFile (1,1) string
    outputFile (1,1) string
end
a = pahybrid.read_acquisition(inputFile);
assert(isequal(size(a.signals), [5,73]), 'PAHybrid:Shape', 'Wrong 5 x 73 signal layout');
assert(isequal(size(a.detectors_m),[5,3]), 'PAHybrid:Shape', 'Wrong 5 x 3 coordinate layout');
assert(abs(a.signals(2,3) - (sin(2*4/17)+0.01+0.0002)) < 1e-13, ...
    'PAHybrid:ChannelOrder', 'Channel/sample indexing mismatch');
assert(abs(a.detectors_m(1,1) - (-.004)) < 1e-14, ...
    'PAHybrid:Geometry', 'Detector order mismatch');
pa_bridge_roundtrip(inputFile,outputFile);
fprintf('PR03 MATLAB transport smoke completed. Verify output in Python.\n');
end
