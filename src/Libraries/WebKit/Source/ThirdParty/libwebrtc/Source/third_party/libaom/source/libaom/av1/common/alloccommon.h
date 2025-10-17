/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 9, 2023.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#ifndef AOM_AV1_COMMON_ALLOCCOMMON_H_
#define AOM_AV1_COMMON_ALLOCCOMMON_H_

#define INVALID_IDX -1  // Invalid buffer index.

#include <stdbool.h>

#include "config/aom_config.h"

#include "av1/common/enums.h"

#ifdef __cplusplus
extern "C" {
#endif

struct AV1Common;
struct BufferPool;
struct CommonContexts;
struct CommonModeInfoParams;
struct AV1CdefWorker;
struct AV1CdefSyncData;

void av1_remove_common(struct AV1Common *cm);

int av1_alloc_above_context_buffers(struct CommonContexts *above_contexts,
                                    int num_tile_rows, int num_mi_cols,
                                    int num_planes);
void av1_free_above_context_buffers(struct CommonContexts *above_contexts);
int av1_alloc_context_buffers(struct AV1Common *cm, int width, int height,
                              BLOCK_SIZE min_partition_size);
void av1_init_mi_buffers(struct CommonModeInfoParams *mi_params);
void av1_free_context_buffers(struct AV1Common *cm);

void av1_free_ref_frame_buffers(struct BufferPool *pool);
void av1_alloc_cdef_buffers(struct AV1Common *const cm,
                            struct AV1CdefWorker **cdef_worker,
                            struct AV1CdefSyncData *cdef_sync, int num_workers,
                            int init_worker);
void av1_free_cdef_buffers(struct AV1Common *const cm,
                           struct AV1CdefWorker **cdef_worker,
                           struct AV1CdefSyncData *cdef_sync);
#if !CONFIG_REALTIME_ONLY || CONFIG_AV1_DECODER
void av1_alloc_restoration_buffers(struct AV1Common *cm, bool is_sgr_enabled);
void av1_free_restoration_buffers(struct AV1Common *cm);
#endif  // !CONFIG_REALTIME_ONLY || CONFIG_AV1_DECODER

int av1_alloc_state_buffers(struct AV1Common *cm, int width, int height);
void av1_free_state_buffers(struct AV1Common *cm);

int av1_get_MBs(int width, int height);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_COMMON_ALLOCCOMMON_H_
