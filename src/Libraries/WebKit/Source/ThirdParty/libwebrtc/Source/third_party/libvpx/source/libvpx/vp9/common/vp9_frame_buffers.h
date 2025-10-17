/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 26, 2025.
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
#ifndef VPX_VP9_COMMON_VP9_FRAME_BUFFERS_H_
#define VPX_VP9_COMMON_VP9_FRAME_BUFFERS_H_

#include "vpx/vpx_frame_buffer.h"
#include "vpx/vpx_integer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct InternalFrameBuffer {
  uint8_t *data;
  size_t size;
  int in_use;
} InternalFrameBuffer;

typedef struct InternalFrameBufferList {
  int num_internal_frame_buffers;
  InternalFrameBuffer *int_fb;
} InternalFrameBufferList;

// Initializes |list|. Returns 0 on success.
int vp9_alloc_internal_frame_buffers(InternalFrameBufferList *list);

// Free any data allocated to the frame buffers.
void vp9_free_internal_frame_buffers(InternalFrameBufferList *list);

// Callback used by libvpx to request an external frame buffer. |cb_priv|
// Callback private data, which points to an InternalFrameBufferList.
// |min_size| is the minimum size in bytes needed to decode the next frame.
// |fb| pointer to the frame buffer.
int vp9_get_frame_buffer(void *cb_priv, size_t min_size,
                         vpx_codec_frame_buffer_t *fb);

// Callback used by libvpx when there are no references to the frame buffer.
// |cb_priv| is not used. |fb| pointer to the frame buffer.
int vp9_release_frame_buffer(void *cb_priv, vpx_codec_frame_buffer_t *fb);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP9_COMMON_VP9_FRAME_BUFFERS_H_
