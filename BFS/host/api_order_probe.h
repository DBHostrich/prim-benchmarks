#ifndef _BFS_API_ORDER_PROBE_H_
#define _BFS_API_ORDER_PROBE_H_

#include <dpu.h>

#include <stdbool.h>
#include <stdint.h>

#include "../support/common.h"

bool bfsApiOrderProbeRequested(void);

bool bfsRunApiOrderProbe(
    struct dpu_set_t dpuSet,
    uint32_t configuredDpus,
    const struct DPUParams* dpuParams,
    const uint32_t* dpuParamsM,
    uint32_t numNodes
);

#endif
