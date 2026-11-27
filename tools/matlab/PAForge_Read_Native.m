function report = PAForge_Read_Native(sampleFolder)
% PAForge_Read_Native  Read a PAForge Python-generated native HDF5 sample.
% This is an interoperability check, NOT verified PATBox algorithm parity.
% Usage: report = PAForge_Read_Native('/absolute/path/material_phantom_00000');
arguments
    sampleFolder (1,1) string
end
rawFile = fullfile(sampleFolder, "raw_signals.h5");
truthFile = fullfile(sampleFolder, "ground_truth.h5");
reconFile = fullfile(sampleFolder, "reconstruction.h5");
assert(isfile(rawFile) && isfile(truthFile) && isfile(reconFile), ...
    "Expected raw_signals.h5, ground_truth.h5, reconstruction.h5");
raw = h5read(rawFile, '/raw_data');
p0 = h5read(truthFile, '/initial_pressure_pa');
recon = h5read(reconFile, '/image');
assert(all(isfinite(raw(:))) && all(isfinite(p0(:))) && all(isfinite(recon(:))), ...
    'Nonfinite data encountered');
report = struct('raw_size', size(raw), 'p0_size', size(p0), ...
    'reconstruction_size', size(recon), 'provenance', 'SYNTHETIC');
disp(report);
figure('Name','PAForge MATLAB/HDF5 readback (synthetic)');
tiledlayout(1,3);
nexttile; imagesc(p0); axis image; colorbar; title('Synthetic p0');
nexttile; imagesc(raw); colorbar; title('Synthetic detector data');
nexttile; imagesc(recon); axis image; colorbar; title('Reconstruction');
end
