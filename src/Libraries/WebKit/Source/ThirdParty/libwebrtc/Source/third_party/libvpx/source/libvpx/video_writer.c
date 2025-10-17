/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 28, 2024.
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

#include "./ivfenc.h"
#include "./video_writer.h"
#include "vpx/vpx_encoder.h"

struct VpxVideoWriterStruct {
  VpxVideoInfo info;
  FILE *file;
  int frame_count;
};

static void write_header(FILE *file, const VpxVideoInfo *info,
                         int frame_count) {
  struct vpx_codec_enc_cfg cfg;
  cfg.g_w = info->frame_width;
  cfg.g_h = info->frame_height;
  cfg.g_timebase.num = info->time_base.numerator;
  cfg.g_timebase.den = info->time_base.denominator;

  ivf_write_file_header(file, &cfg, info->codec_fourcc, frame_count);
}

VpxVideoWriter *vpx_video_writer_open(const char *filename,
                                      VpxContainer container,
                                      const VpxVideoInfo *info) {
  if (container == kContainerIVF) {
    VpxVideoWriter *writer = NULL;
    FILE *const file = fopen(filename, "wb");
    if (!file) {
      fprintf(stderr, "%s can't be written to.\n", filename);
      return NULL;
    }
    writer = malloc(sizeof(*writer));
    if (!writer) {
      fprintf(stderr, "Can't allocate VpxVideoWriter.\n");
      return NULL;
    }
    writer->frame_count = 0;
    writer->info = *info;
    writer->file = file;

    write_header(writer->file, info, 0);

    return writer;
  }
  fprintf(stderr, "VpxVideoWriter supports only IVF.\n");
  return NULL;
}

void vpx_video_writer_close(VpxVideoWriter *writer) {
  if (writer) {
    // Rewriting frame header with real frame count
    rewind(writer->file);
    write_header(writer->file, &writer->info, writer->frame_count);

    fclose(writer->file);
    free(writer);
  }
}

int vpx_video_writer_write_frame(VpxVideoWriter *writer, const uint8_t *buffer,
                                 size_t size, int64_t pts) {
  ivf_write_frame_header(writer->file, pts, size);
  if (fwrite(buffer, 1, size, writer->file) != size) return 0;

  ++writer->frame_count;

  return 1;
}
