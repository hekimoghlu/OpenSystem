/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 13, 2022.
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
#include "libyuv/rotate_row.h"
#include "libyuv/row.h"

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

void TransposeWx8_C(const uint8_t* src,
                    int src_stride,
                    uint8_t* dst,
                    int dst_stride,
                    int width) {
  int i;
  for (i = 0; i < width; ++i) {
    dst[0] = src[0 * src_stride];
    dst[1] = src[1 * src_stride];
    dst[2] = src[2 * src_stride];
    dst[3] = src[3 * src_stride];
    dst[4] = src[4 * src_stride];
    dst[5] = src[5 * src_stride];
    dst[6] = src[6 * src_stride];
    dst[7] = src[7 * src_stride];
    ++src;
    dst += dst_stride;
  }
}

void TransposeUVWx8_C(const uint8_t* src,
                      int src_stride,
                      uint8_t* dst_a,
                      int dst_stride_a,
                      uint8_t* dst_b,
                      int dst_stride_b,
                      int width) {
  int i;
  for (i = 0; i < width; ++i) {
    dst_a[0] = src[0 * src_stride + 0];
    dst_b[0] = src[0 * src_stride + 1];
    dst_a[1] = src[1 * src_stride + 0];
    dst_b[1] = src[1 * src_stride + 1];
    dst_a[2] = src[2 * src_stride + 0];
    dst_b[2] = src[2 * src_stride + 1];
    dst_a[3] = src[3 * src_stride + 0];
    dst_b[3] = src[3 * src_stride + 1];
    dst_a[4] = src[4 * src_stride + 0];
    dst_b[4] = src[4 * src_stride + 1];
    dst_a[5] = src[5 * src_stride + 0];
    dst_b[5] = src[5 * src_stride + 1];
    dst_a[6] = src[6 * src_stride + 0];
    dst_b[6] = src[6 * src_stride + 1];
    dst_a[7] = src[7 * src_stride + 0];
    dst_b[7] = src[7 * src_stride + 1];
    src += 2;
    dst_a += dst_stride_a;
    dst_b += dst_stride_b;
  }
}

void TransposeWxH_C(const uint8_t* src,
                    int src_stride,
                    uint8_t* dst,
                    int dst_stride,
                    int width,
                    int height) {
  int i;
  for (i = 0; i < width; ++i) {
    int j;
    for (j = 0; j < height; ++j) {
      dst[i * dst_stride + j] = src[j * src_stride + i];
    }
  }
}

void TransposeUVWxH_C(const uint8_t* src,
                      int src_stride,
                      uint8_t* dst_a,
                      int dst_stride_a,
                      uint8_t* dst_b,
                      int dst_stride_b,
                      int width,
                      int height) {
  int i;
  for (i = 0; i < width * 2; i += 2) {
    int j;
    for (j = 0; j < height; ++j) {
      dst_a[j + ((i >> 1) * dst_stride_a)] = src[i + (j * src_stride)];
      dst_b[j + ((i >> 1) * dst_stride_b)] = src[i + (j * src_stride) + 1];
    }
  }
}

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif
