#!/usr/bin/env bash
set -euo pipefail
paforge research-sources
paforge datasets download --dataset digital-twin-iqa --profile starter --accept-licenses --max-file-gb 0.2 --max-total-gb 0.3 --output data/external
paforge datasets download --dataset openqpai --profile starter --accept-licenses --max-file-gb 0.5 --max-total-gb 0.6 --output data/external
if [[ -f data/external/digital-twin-iqa/ipasc_file.hdf5 ]]; then
  paforge inventory-hdf5 data/external/digital-twin-iqa/ipasc_file.hdf5 --output experiments/ipasc_inventory.json
fi
echo 'Dataset files acquired if previous commands succeeded. Review schema before calibration or reconstruction.'
