#!/bin/sh

# Directory to this SH file.
BASE_DIR=$(dirname "$0")

PYTHON_BIN="$BASE_DIR/5.2/python/bin/python3.13"
SYSTEM_INFO_STARTUP_PY="$BASE_DIR/5.2/scripts/modules/_bpy_internal/system_info/url_prefill_startup.py"
if test -f "$PYTHON_BIN"; then
  exec "$PYTHON_BIN" -I "$SYSTEM_INFO_STARTUP_PY"
fi

echo "ERROR: Failed to find Python executable at: $PYTHON_BIN"
echo "Possible causes include:"
echo "- Your Blender installation is corrupt or missing python."
echo "- The location or name of the python executable has changed."
exit 1
