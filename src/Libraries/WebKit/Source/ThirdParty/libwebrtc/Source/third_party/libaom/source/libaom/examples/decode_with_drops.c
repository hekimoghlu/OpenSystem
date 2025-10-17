/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 27, 2023.
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
// Decode With Drops Example
// =========================
//
// This is an example utility which drops a series of frames, as specified
// on the command line. This is useful for observing the error recovery
// features of the codec.
//
// Usage
// -----
// This example adds a single argument to the `simple_decoder` example,
// which specifies the range or pattern of frames to drop. The parameter is
// parsed as follows:
//
// Dropping A Range Of Frames
// --------------------------
// To drop a range of frames, specify the starting frame and the ending
// frame to drop, separated by a dash. The following command will drop
// frames 5 through 10 (base 1).
//
//  $ ./decode_with_drops in.ivf out.i420 5-10
//
//
// Dropping A Pattern Of Frames
// ----------------------------
// To drop a pattern of frames, specify the number of frames to drop and
// the number of frames after which to repeat the pattern, separated by
// a forward-slash. The following command will drop 3 of 7 frames.
// Specifically, it will decode 4 frames, then drop 3 frames, and then
// repeat.
//
//  $ ./decode_with_drops in.ivf out.i420 3/7
//
//
// Extra Variables
// ---------------
// This example maintains the pattern passed on the command line in the
// `n`, `m`, and `is_range` variables:
//
//
// Making The Drop Decision
// ------------------------
// The example decides whether to drop the frame based on the current
// frame number, immediately before decoding the frame.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "aom/aom_decoder.h"
#include "aom/aomdx.h"
#include "common/tools_common.h"
#include "common/video_reader.h"

static const char *exec_name;

void usage_exit(void) {
  fprintf(stderr, "Usage: %s <infile> <outfile> <N-M|N/M>\n", exec_name);
  exit(EXIT_FAILURE);
}

int main(int argc, char **argv) {
  int frame_cnt = 0;
  FILE *outfile = NULL;
  AvxVideoReader *reader = NULL;
  const AvxVideoInfo *info = NULL;
  int n = 0;
  int m = 0;
  int is_range = 0;
  char *nptr = NULL;

  exec_name = argv[0];

  if (argc != 4) die("Invalid number of arguments.");

  reader = aom_video_reader_open(argv[1]);
  if (!reader) die("Failed to open %s for reading.", argv[1]);

  if (!(outfile = fopen(argv[2], "wb")))
    die("Failed to open %s for writing.", argv[2]);

  n = (int)strtol(argv[3], &nptr, 0);
  m = (int)strtol(nptr + 1, NULL, 0);
  is_range = (*nptr == '-');
  if (!n || !m || (*nptr != '-' && *nptr != '/'))
    die("Couldn't parse pattern %s.\n", argv[3]);

  info = aom_video_reader_get_info(reader);

  aom_codec_iface_t *decoder = get_aom_decoder_by_fourcc(info->codec_fourcc);
  if (!decoder) die("Unknown input codec.");

  printf("Using %s\n", aom_codec_iface_name(decoder));
  aom_codec_ctx_t codec;
  if (aom_codec_dec_init(&codec, decoder, NULL, 0))
    die("Failed to initialize decoder.");

  while (aom_video_reader_read_frame(reader)) {
    aom_codec_iter_t iter = NULL;
    aom_image_t *img = NULL;
    size_t frame_size = 0;
    int skip;
    const unsigned char *frame =
        aom_video_reader_get_frame(reader, &frame_size);
    ++frame_cnt;

    skip = (is_range && frame_cnt >= n && frame_cnt <= m) ||
           (!is_range && m - (frame_cnt - 1) % m <= n);

    if (!skip) {
      putc('.', stdout);
      if (aom_codec_decode(&codec, frame, frame_size, NULL))
        die_codec(&codec, "Failed to decode frame.");

      while ((img = aom_codec_get_frame(&codec, &iter)) != NULL)
        aom_img_write(img, outfile);
    } else {
      putc('X', stdout);
    }

    fflush(stdout);
  }

  printf("Processed %d frames.\n", frame_cnt);
  if (aom_codec_destroy(&codec)) die_codec(&codec, "Failed to destroy codec.");

  printf("Play: ffplay -f rawvideo -pix_fmt yuv420p -s %dx%d %s\n",
         info->frame_width, info->frame_height, argv[2]);

  aom_video_reader_close(reader);
  fclose(outfile);

  return EXIT_SUCCESS;
}
