/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 27, 2024.
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

#include "vpx/vp8dx.h"
#include "vpx/vpx_decoder.h"

#include "../tools_common.h"
#include "../video_reader.h"
#include "./vpx_config.h"

static const char *exec_name;

void usage_exit(void) {
  fprintf(stderr, "Usage: %s <infile> <outfile> <N-M|N/M>\n", exec_name);
  exit(EXIT_FAILURE);
}

int main(int argc, char **argv) {
  int frame_cnt = 0;
  FILE *outfile = NULL;
  vpx_codec_ctx_t codec;
  const VpxInterface *decoder = NULL;
  VpxVideoReader *reader = NULL;
  const VpxVideoInfo *info = NULL;
  int n = 0;
  int m = 0;
  int is_range = 0;
  char *nptr = NULL;

  exec_name = argv[0];

  if (argc != 4) die("Invalid number of arguments.");

  reader = vpx_video_reader_open(argv[1]);
  if (!reader) die("Failed to open %s for reading.", argv[1]);

  if (!(outfile = fopen(argv[2], "wb")))
    die("Failed to open %s for writing.", argv[2]);

  n = (int)strtol(argv[3], &nptr, 0);
  m = (int)strtol(nptr + 1, NULL, 0);
  is_range = (*nptr == '-');
  if (!n || !m || (*nptr != '-' && *nptr != '/'))
    die("Couldn't parse pattern %s.\n", argv[3]);

  info = vpx_video_reader_get_info(reader);

  decoder = get_vpx_decoder_by_fourcc(info->codec_fourcc);
  if (!decoder) die("Unknown input codec.");

  printf("Using %s\n", vpx_codec_iface_name(decoder->codec_interface()));

  if (vpx_codec_dec_init(&codec, decoder->codec_interface(), NULL, 0))
    die("Failed to initialize decoder.");

  while (vpx_video_reader_read_frame(reader)) {
    vpx_codec_iter_t iter = NULL;
    vpx_image_t *img = NULL;
    size_t frame_size = 0;
    int skip;
    const unsigned char *frame =
        vpx_video_reader_get_frame(reader, &frame_size);
    if (vpx_codec_decode(&codec, frame, (unsigned int)frame_size, NULL, 0))
      die_codec(&codec, "Failed to decode frame.");

    ++frame_cnt;

    skip = (is_range && frame_cnt >= n && frame_cnt <= m) ||
           (!is_range && m - (frame_cnt - 1) % m <= n);

    if (!skip) {
      putc('.', stdout);

      while ((img = vpx_codec_get_frame(&codec, &iter)) != NULL)
        vpx_img_write(img, outfile);
    } else {
      putc('X', stdout);
    }

    fflush(stdout);
  }

  printf("Processed %d frames.\n", frame_cnt);
  if (vpx_codec_destroy(&codec)) die_codec(&codec, "Failed to destroy codec.");

  printf("Play: ffplay -f rawvideo -pix_fmt yuv420p -s %dx%d %s\n",
         info->frame_width, info->frame_height, argv[2]);

  vpx_video_reader_close(reader);
  fclose(outfile);

  return EXIT_SUCCESS;
}
