/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 28, 2022.
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
#ifndef VPX_VP9_COMMON_VP9_ALLOCCOMMON_H_
#define VPX_VP9_COMMON_VP9_ALLOCCOMMON_H_

#define INVALID_IDX (-1)  // Invalid buffer index.

#ifdef __cplusplus
extern "C" {
#endif

struct VP9Common;
struct BufferPool;

void vp9_remove_common(struct VP9Common *cm);

int vp9_alloc_loop_filter(struct VP9Common *cm);
int vp9_alloc_context_buffers(struct VP9Common *cm, int width, int height);
void vp9_init_context_buffers(struct VP9Common *cm);
void vp9_free_context_buffers(struct VP9Common *cm);

void vp9_free_ref_frame_buffers(struct BufferPool *pool);
void vp9_free_postproc_buffers(struct VP9Common *cm);

int vp9_alloc_state_buffers(struct VP9Common *cm, int width, int height);
void vp9_free_state_buffers(struct VP9Common *cm);

void vp9_set_mi_size(int *mi_rows, int *mi_cols, int *mi_stride, int width,
                     int height);
void vp9_set_mb_size(int *mb_rows, int *mb_cols, int *mb_num, int mi_rows,
                     int mi_cols);

void vp9_set_mb_mi(struct VP9Common *cm, int width, int height);

void vp9_swap_current_and_last_seg_map(struct VP9Common *cm);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP9_COMMON_VP9_ALLOCCOMMON_H_
