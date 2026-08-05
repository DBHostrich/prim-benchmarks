#ifndef _BFS_CONTEXT_PROBE_H_
#define _BFS_CONTEXT_PROBE_H_

#include <dpu.h>

#include <stdbool.h>
#include <stdint.h>

#include "../support/common.h"

bool bfsContextProbeRequested(void);

bool bfsRunContextProbe(
    struct dpu_set_t dpuSet,
    uint32_t configuredDpus,
    const struct DPUParams* dpuParams,
    uint32_t numNodes
);

#endif
