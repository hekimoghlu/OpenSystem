/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 14, 2023.
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
#include <stdlib.h>
#include <string.h>

#include "./ivfdec.h"
#include "./video_reader.h"

#include "vpx_ports/mem_ops.h"

static const char *const kIVFSignature = "DKIF";

struct VpxVideoReaderStruct {
  VpxVideoInfo info;
  FILE *file;
  uint8_t *buffer;
  size_t buffer_size;
  size_t frame_size;
};

VpxVideoReader *vpx_video_reader_open(const char *filename) {
  char header[32];
  VpxVideoReader *reader = NULL;
  FILE *const file = fopen(filename, "rb");
  if (!file) {
    fprintf(stderr, "%s can't be opened.\n", filename);  // Can't open file
    return NULL;
  }

  if (fread(header, 1, 32, file) != 32) {
    fprintf(stderr, "File header on %s can't be read.\n",
            filename);  // Can't read file header
    return NULL;
  }
  if (memcmp(kIVFSignature, header, 4) != 0) {
    fprintf(stderr, "The IVF signature on %s is wrong.\n",
            filename);  // Wrong IVF signature

    return NULL;
  }
  if (mem_get_le16(header + 4) != 0) {
    fprintf(stderr, "%s uses the wrong IVF version.\n",
            filename);  // Wrong IVF version

    return NULL;
  }

  reader = calloc(1, sizeof(*reader));
  if (!reader) {
    fprintf(
        stderr,
        "Can't allocate VpxVideoReader\n");  // Can't allocate VpxVideoReader

    return NULL;
  }

  reader->file = file;
  reader->info.codec_fourcc = mem_get_le32(header + 8);
  reader->info.frame_width = mem_get_le16(header + 12);
  reader->info.frame_height = mem_get_le16(header + 14);
  reader->info.time_base.numerator = mem_get_le32(header + 16);
  reader->info.time_base.denominator = mem_get_le32(header + 20);

  return reader;
}

void vpx_video_reader_close(VpxVideoReader *reader) {
  if (reader) {
    fclose(reader->file);
    free(reader->buffer);
    free(reader);
  }
}

int vpx_video_reader_read_frame(VpxVideoReader *reader) {
  return !ivf_read_frame(reader->file, &reader->buffer, &reader->frame_size,
                         &reader->buffer_size);
}

const uint8_t *vpx_video_reader_get_frame(VpxVideoReader *reader,
                                          size_t *size) {
  if (size) *size = reader->frame_size;

  return reader->buffer;
}

const VpxVideoInfo *vpx_video_reader_get_info(VpxVideoReader *reader) {
  return &reader->info;
}
