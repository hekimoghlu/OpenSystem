/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 20, 2024.
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
#ifndef AOM_COMMON_VIDEO_WRITER_H_
#define AOM_COMMON_VIDEO_WRITER_H_

#include "common/video_common.h"

enum { kContainerIVF } UENUM1BYTE(AvxContainer);

struct AvxVideoWriterStruct;
typedef struct AvxVideoWriterStruct AvxVideoWriter;

#ifdef __cplusplus
extern "C" {
#endif

// Finds and opens writer for specified container format.
// Returns an opaque AvxVideoWriter* upon success, or NULL upon failure.
// Right now only IVF format is supported.
AvxVideoWriter *aom_video_writer_open(const char *filename,
                                      AvxContainer container,
                                      const AvxVideoInfo *info);

// Frees all resources associated with AvxVideoWriter* returned from
// aom_video_writer_open() call.
void aom_video_writer_close(AvxVideoWriter *writer);

// Writes frame bytes to the file.
int aom_video_writer_write_frame(AvxVideoWriter *writer, const uint8_t *buffer,
                                 size_t size, int64_t pts);
// Set fourcc.
void aom_video_writer_set_fourcc(AvxVideoWriter *writer, uint32_t fourcc);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_COMMON_VIDEO_WRITER_H_
