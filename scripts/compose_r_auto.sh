#!/usr/bin/env bash
find -name "render_*" -type d -exec ../../scripts/compose_r.sh {} \;

