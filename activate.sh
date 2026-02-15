#!/bin/bash

source .venv/bin/activate

export MOLMO_DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/data"
