/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 21, 2025.
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
#ifndef AOM_COMMON_VIDEO_READER_H_
#define AOM_COMMON_VIDEO_READER_H_

#include "common/video_common.h"

// The following code is work in progress. It is going to  support transparent
// reading of input files. Right now only IVF format is supported for
// simplicity. The main goal the API is to be simple and easy to use in example
// code and in aomenc/aomdec later. All low-level details like memory
// buffer management are hidden from API users.
struct AvxVideoReaderStruct;
typedef struct AvxVideoReaderStruct AvxVideoReader;

#ifdef __cplusplus
extern "C" {
#endif

// Opens the input file for reading and inspects it to determine file type.
// Returns an opaque AvxVideoReader* upon success, or NULL upon failure.
// Right now only IVF format is supported.
AvxVideoReader *aom_video_reader_open(const char *filename);

// Frees all resources associated with AvxVideoReader* returned from
// aom_video_reader_open() call.
void aom_video_reader_close(AvxVideoReader *reader);

// Reads frame from the file and stores it in internal buffer.
int aom_video_reader_read_frame(AvxVideoReader *reader);

// Returns the pointer to memory buffer with frame data read by last call to
// aom_video_reader_read_frame().
const uint8_t *aom_video_reader_get_frame(AvxVideoReader *reader, size_t *size);

// Returns the pts of the frame.
int64_t aom_video_reader_get_frame_pts(AvxVideoReader *reader);
// Return the reader file.
FILE *aom_video_reader_get_file(AvxVideoReader *reader);

// Fills AvxVideoInfo with information from opened video file.
const AvxVideoInfo *aom_video_reader_get_info(AvxVideoReader *reader);

// Set fourcc.
void aom_video_reader_set_fourcc(AvxVideoReader *reader, uint32_t fourcc);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_COMMON_VIDEO_READER_H_
