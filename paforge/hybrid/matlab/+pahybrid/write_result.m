function write_result(filePath, image, algorithm, acquisitionID, kgrid, imageGrid)
%WRITE_RESULT Save a PATBox image in row-major flattened exchange representation.
% This is a standalone output and does NOT claim physical quantitative units.
arguments
    filePath (1,1) string
    image {mustBeNumeric}
    algorithm (1,1) string
    acquisitionID (1,1) string
    kgrid
    imageGrid (1,1) struct = struct()
end
assert(ismatrix(image) && ~isempty(image) && isreal(image) && ...
       all(isfinite(image(:))), 'PAHybrid:Image', 'Expected finite real image');
assert(~isfile(filePath), 'PAHybrid:Exists', 'Refusing to overwrite reconstruction');
image = double(image);
[nr,nc] = size(image);
% image.'(:) == flatten C-order of Python [height,width].
flat = reshape(image.', [], 1);
h5create(filePath, '/reconstruction/image_flat_c', [numel(flat),1], 'Datatype', 'double');
h5write(filePath, '/reconstruction/image_flat_c', flat);
h5writeatt(filePath, '/', 'format', 'PA-HYBRID-RECONSTRUCTION');
h5writeatt(filePath, '/', 'schema_version', '0.1.0');
h5writeatt(filePath, '/reconstruction', 'height', int64(nr));
h5writeatt(filePath, '/reconstruction', 'width', int64(nc));
h5writeatt(filePath, '/reconstruction', 'method_id', char("PATBox:" + algorithm));
h5writeatt(filePath, '/reconstruction', 'acquisition_id', char(acquisitionID));
h5writeatt(filePath, '/reconstruction', 'quantitative', int64(0));
h5writeatt(filePath, '/reconstruction', 'signal_units', 'not calibrated');
if isprop(kgrid, 'dx') && isprop(kgrid, 'dy')
    h5writeatt(filePath, '/reconstruction', 'grid_dx_m', double(kgrid.dx));
    h5writeatt(filePath, '/reconstruction', 'grid_dy_m', double(kgrid.dy));
end

% PR-03 parity is permitted ONLY with an independently reviewed image grid.
if ~isempty(fieldnames(imageGrid))
    f = {'dx_m','dy_m','origin_x_m','origin_y_m','orientation'};
    assert(all(isfield(imageGrid, f)), 'PAHybrid:Grid', ...
        'Explicit calibrated image grid requires all five attributes');
    assert(isfinite(imageGrid.dx_m) && imageGrid.dx_m>0 && ...
        isfinite(imageGrid.dy_m) && imageGrid.dy_m>0 && ...
        isfinite(imageGrid.origin_x_m) && isfinite(imageGrid.origin_y_m) && ...
        strlength(string(imageGrid.orientation))>0, 'PAHybrid:Grid', ...
        'Nonfinite/unknown image alignment');
    h5writeatt(filePath, '/reconstruction', 'grid_dx_m', double(imageGrid.dx_m));
    h5writeatt(filePath, '/reconstruction', 'grid_dy_m', double(imageGrid.dy_m));
    h5writeatt(filePath, '/reconstruction', 'grid_origin_x_m', double(imageGrid.origin_x_m));
    h5writeatt(filePath, '/reconstruction', 'grid_origin_y_m', double(imageGrid.origin_y_m));
    h5writeatt(filePath, '/reconstruction', 'grid_orientation', char(string(imageGrid.orientation)));
end

end
