/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 18, 2024.
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
#include <math.h>
#include <stdlib.h>

#include "./vpx_config.h"
#include "./vpx_dsp_rtcd.h"

#include "vpx/vpx_integer.h"
#include "vpx_dsp/postproc.h"
#include "vpx_ports/mem.h"

void vpx_plane_add_noise_c(uint8_t *start, const int8_t *noise, int blackclamp,
                           int whiteclamp, int width, int height, int pitch) {
  int i, j;
  int bothclamp = blackclamp + whiteclamp;
  for (i = 0; i < height; ++i) {
    uint8_t *pos = start + i * pitch;
    const int8_t *ref = (const int8_t *)(noise + (rand() & 0xff));  // NOLINT

    for (j = 0; j < width; ++j) {
      int v = pos[j];

      v = clamp(v - blackclamp, 0, 255);
      v = clamp(v + bothclamp, 0, 255);
      v = clamp(v - whiteclamp, 0, 255);

      pos[j] = v + ref[j];
    }
  }
}

static double gaussian(double sigma, double mu, double x) {
  return 1 / (sigma * sqrt(2.0 * 3.14159265)) *
         (exp(-(x - mu) * (x - mu) / (2 * sigma * sigma)));
}

int vpx_setup_noise(double sigma, int8_t *noise, int size) {
  int8_t char_dist[256];
  int next = 0, i, j;

  // set up a 256 entry lookup that matches gaussian distribution
  for (i = -32; i < 32; ++i) {
    const int a_i = (int)(0.5 + 256 * gaussian(sigma, 0, i));
    if (a_i) {
      for (j = 0; j < a_i; ++j) {
        if (next + j >= 256) goto set_noise;
        char_dist[next + j] = (int8_t)i;
      }
      next = next + j;
    }
  }

  // Rounding error - might mean we have less than 256.
  for (; next < 256; ++next) {
    char_dist[next] = 0;
  }

set_noise:
  for (i = 0; i < size; ++i) {
    noise[i] = char_dist[rand() & 0xff];  // NOLINT
  }

  // Returns the highest non 0 value used in distribution.
  return -char_dist[0];
}
