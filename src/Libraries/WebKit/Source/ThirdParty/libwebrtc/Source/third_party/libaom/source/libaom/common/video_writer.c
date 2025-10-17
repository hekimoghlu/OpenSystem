/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 4, 2023.
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
#include "common/video_writer.h"

#include <stdlib.h>

#include "aom/aom_encoder.h"
#include "common/ivfenc.h"

struct AvxVideoWriterStruct {
  AvxVideoInfo info;
  FILE *file;
  int frame_count;
};

static void write_header(FILE *file, const AvxVideoInfo *info,
                         int frame_count) {
  struct aom_codec_enc_cfg cfg;
  cfg.g_w = info->frame_width;
  cfg.g_h = info->frame_height;
  cfg.g_timebase.num = info->time_base.numerator;
  cfg.g_timebase.den = info->time_base.denominator;

  ivf_write_file_header(file, &cfg, info->codec_fourcc, frame_count);
}

AvxVideoWriter *aom_video_writer_open(const char *filename,
                                      AvxContainer container,
                                      const AvxVideoInfo *info) {
  if (container == kContainerIVF) {
    AvxVideoWriter *writer = NULL;
    FILE *const file = fopen(filename, "wb");
    if (!file) return NULL;

    writer = malloc(sizeof(*writer));
    if (!writer) {
      fclose(file);
      return NULL;
    }
    writer->frame_count = 0;
    writer->info = *info;
    writer->file = file;

    write_header(writer->file, info, 0);

    return writer;
  }

  return NULL;
}

void aom_video_writer_close(AvxVideoWriter *writer) {
  if (writer) {
    // Rewriting frame header with real frame count
    rewind(writer->file);
    write_header(writer->file, &writer->info, writer->frame_count);

    fclose(writer->file);
    free(writer);
  }
}

int aom_video_writer_write_frame(AvxVideoWriter *writer, const uint8_t *buffer,
                                 size_t size, int64_t pts) {
  ivf_write_frame_header(writer->file, pts, size);
  if (fwrite(buffer, 1, size, writer->file) != size) return 0;

  ++writer->frame_count;

  return 1;
}

void aom_video_writer_set_fourcc(AvxVideoWriter *writer, uint32_t fourcc) {
  writer->info.codec_fourcc = fourcc;
}
