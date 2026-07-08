#pragma once

#include "common.h"

// Env-gated MoE routing trace. When GGML_MOE_TRACE=<file> is set and the tool
// has not installed its own eval callback, dump per-token per-layer routed
// expert ids to <file> for offline expert-cache simulation. See moe-trace.cpp
// for the binary format.
void common_moe_trace_maybe_install(common_params & params);
