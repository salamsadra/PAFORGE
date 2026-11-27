function ok = pa_bridge_validate(filePath)
%PA_BRIDGE_VALIDATE Validate the exchange shape and basic provenance in MATLAB.
acq = pahybrid.read_acquisition(string(filePath));
assert(isfield(acq.metadata, 'acquisition_id') && ...
       strlength(string(acq.metadata.acquisition_id)) > 0, ...
       'PAHybrid:Provenance', 'An acquisition ID is required');
ok = true;
end
