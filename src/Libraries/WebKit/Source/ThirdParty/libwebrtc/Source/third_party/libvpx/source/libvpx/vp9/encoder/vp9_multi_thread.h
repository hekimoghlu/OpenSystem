/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 11, 2023.
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
#ifndef VPX_VP9_ENCODER_VP9_MULTI_THREAD_H_
#define VPX_VP9_ENCODER_VP9_MULTI_THREAD_H_

#include "vp9/encoder/vp9_encoder.h"
#include "vp9/encoder/vp9_job_queue.h"

void *vp9_enc_grp_get_next_job(MultiThreadHandle *multi_thread_ctxt,
                               int tile_id);

void vp9_prepare_job_queue(VP9_COMP *cpi, JOB_TYPE job_type);

int vp9_get_job_queue_status(MultiThreadHandle *multi_thread_ctxt,
                             int cur_tile_id);

void vp9_assign_tile_to_thread(MultiThreadHandle *multi_thread_ctxt,
                               int tile_cols, int num_workers);

void vp9_multi_thread_tile_init(VP9_COMP *cpi);

void vp9_row_mt_mem_alloc(VP9_COMP *cpi);

void vp9_row_mt_alloc_rd_thresh(VP9_COMP *const cpi,
                                TileDataEnc *const this_tile);

void vp9_row_mt_mem_dealloc(VP9_COMP *cpi);

int vp9_get_tiles_proc_status(MultiThreadHandle *multi_thread_ctxt,
                              int *tile_completion_status, int *cur_tile_id,
                              int tile_cols);

#endif  // VPX_VP9_ENCODER_VP9_MULTI_THREAD_H_
