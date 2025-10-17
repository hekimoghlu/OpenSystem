/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 20, 2024.
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
#ifndef VPX_VIDEO_READER_H_
#define VPX_VIDEO_READER_H_

#include "./video_common.h"

// The following code is work in progress. It is going to  support transparent
// reading of input files. Right now only IVF format is supported for
// simplicity. The main goal the API is to be simple and easy to use in example
// code and in vpxenc/vpxdec later. All low-level details like memory
// buffer management are hidden from API users.
struct VpxVideoReaderStruct;
typedef struct VpxVideoReaderStruct VpxVideoReader;

#ifdef __cplusplus
extern "C" {
#endif

// Opens the input file for reading and inspects it to determine file type.
// Returns an opaque VpxVideoReader* upon success, or NULL upon failure.
// Right now only IVF format is supported.
VpxVideoReader *vpx_video_reader_open(const char *filename);

// Frees all resources associated with VpxVideoReader* returned from
// vpx_video_reader_open() call.
void vpx_video_reader_close(VpxVideoReader *reader);

// Reads frame from the file and stores it in internal buffer.
int vpx_video_reader_read_frame(VpxVideoReader *reader);

// Returns the pointer to memory buffer with frame data read by last call to
// vpx_video_reader_read_frame().
const uint8_t *vpx_video_reader_get_frame(VpxVideoReader *reader, size_t *size);

// Fills VpxVideoInfo with information from opened video file.
const VpxVideoInfo *vpx_video_reader_get_info(VpxVideoReader *reader);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VIDEO_READER_H_
