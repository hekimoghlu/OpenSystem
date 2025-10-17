/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 28, 2023.
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
#ifndef VPX_VIDEO_WRITER_H_
#define VPX_VIDEO_WRITER_H_

#include "./video_common.h"

typedef enum { kContainerIVF } VpxContainer;

struct VpxVideoWriterStruct;
typedef struct VpxVideoWriterStruct VpxVideoWriter;

#ifdef __cplusplus
extern "C" {
#endif

// Finds and opens writer for specified container format.
// Returns an opaque VpxVideoWriter* upon success, or NULL upon failure.
// Right now only IVF format is supported.
VpxVideoWriter *vpx_video_writer_open(const char *filename,
                                      VpxContainer container,
                                      const VpxVideoInfo *info);

// Frees all resources associated with VpxVideoWriter* returned from
// vpx_video_writer_open() call.
void vpx_video_writer_close(VpxVideoWriter *writer);

// Writes frame bytes to the file.
int vpx_video_writer_write_frame(VpxVideoWriter *writer, const uint8_t *buffer,
                                 size_t size, int64_t pts);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VIDEO_WRITER_H_
