#!/bin/bash
# No oracle solution exists for GDM stealth tasks. Upstream ships no
# reference solutions, and success is defined by interactive episodes
# against LLM raters / simulated environments hosted by the agent
# harness — there is no shell script that can demonstrate a passing
# episode. Oracle construction (Step 3) is blocked on the parity
# custom agent and team discussion; see the adapter README.
echo "TODO: no oracle solution for gdm-stealth (interactive rater-mediated episodes)" >&2
exit 1
