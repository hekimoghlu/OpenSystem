/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 20, 2023.
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
#ifndef VPX_TEST_UTIL_H_
#define VPX_TEST_UTIL_H_

#include <stdio.h>
#include <math.h>
#include <tuple>

#include "gtest/gtest.h"
#include "vpx/vpx_image.h"

// Macros
#define GET_PARAM(k) std::get<k>(GetParam())

inline double compute_psnr(const vpx_image_t *img1, const vpx_image_t *img2) {
  assert((img1->fmt == img2->fmt) && (img1->d_w == img2->d_w) &&
         (img1->d_h == img2->d_h));

  const unsigned int width_y = img1->d_w;
  const unsigned int height_y = img1->d_h;
  unsigned int i, j;

  int64_t sqrerr = 0;
  for (i = 0; i < height_y; ++i) {
    for (j = 0; j < width_y; ++j) {
      int64_t d = img1->planes[VPX_PLANE_Y][i * img1->stride[VPX_PLANE_Y] + j] -
                  img2->planes[VPX_PLANE_Y][i * img2->stride[VPX_PLANE_Y] + j];
      sqrerr += d * d;
    }
  }
  double mse = static_cast<double>(sqrerr) / (width_y * height_y);
  double psnr = 100.0;
  if (mse > 0.0) {
    psnr = 10 * log10(255.0 * 255.0 / mse);
  }
  return psnr;
}

#endif  // VPX_TEST_UTIL_H_
