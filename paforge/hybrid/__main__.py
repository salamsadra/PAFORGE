"""Small CLI for the optional hybrid exchange; no MATLAB needed to inspect files."""
import argparse
import json
from .exchange import inspect_exchange
from .legacy import from_native_hdf5
from .exchange import write_acquisition


def main():
    parser = argparse.ArgumentParser(prog="python -m paforge.hybrid")
    sub = parser.add_subparsers(dest="cmd", required=True)
    inspect_p = sub.add_parser("inspect", help="Validate and summarize a hybrid HDF5")
    inspect_p.add_argument("path")
    convert = sub.add_parser("convert-native", help="Convert an explicitly annotated legacy acquisition")
    convert.add_argument("input")
    convert.add_argument("output")
    for name in ("acquisition-id", "device-id", "session-id", "source-id", "coordinate-frame"):
        convert.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    if args.cmd == "inspect":
        print(json.dumps(inspect_exchange(args.path), indent=2))
    else:
        record = from_native_hdf5(args.input, acquisition_id=args.acquisition_id,
                                  device_id=args.device_id, session_id=args.session_id,
                                  source_id=args.source_id, coordinate_frame=args.coordinate_frame)
        print(write_acquisition(args.output, record))

if __name__ == "__main__":
    main()
