#!/bin/bash
# Oracle candidate for this GDM self-reasoning task.
# Upstream ships no oracle solutions; this scripted candidate targets
# the upstream success condition (see tests/verify.py). It has NOT yet
# passed Step 3 oracle verification (100% reward under Harbor) — see
# the adapter README, Notes & Caveats.
set -e
cd /home/agent
{oracle_body}
