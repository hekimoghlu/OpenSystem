/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 17, 2022.
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
#ifndef INCLUDE_LIBYUV_SCALE_H_
#define INCLUDE_LIBYUV_SCALE_H_

#include "libyuv/basic_types.h"

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

// Supported filtering.
typedef enum FilterMode {
  kFilterNone = 0,      // Point sample; Fastest.
  kFilterLinear = 1,    // Filter horizontally only.
  kFilterBilinear = 2,  // Faster than box, but lower quality scaling down.
  kFilterBox = 3        // Highest quality.
} FilterModeEnum;

// Scale a YUV plane.
LIBYUV_API
void ScalePlane(const uint8_t* src,
                int src_stride,
                int src_width,
                int src_height,
                uint8_t* dst,
                int dst_stride,
                int dst_width,
                int dst_height,
                enum FilterMode filtering);

LIBYUV_API
void ScalePlane_16(const uint16_t* src,
                   int src_stride,
                   int src_width,
                   int src_height,
                   uint16_t* dst,
                   int dst_stride,
                   int dst_width,
                   int dst_height,
                   enum FilterMode filtering);

// Scales a YUV 4:2:0 image from the src width and height to the
// dst width and height.
// If filtering is kFilterNone, a simple nearest-neighbor algorithm is
// used. This produces basic (blocky) quality at the fastest speed.
// If filtering is kFilterBilinear, interpolation is used to produce a better
// quality image, at the expense of speed.
// If filtering is kFilterBox, averaging is used to produce ever better
// quality image, at further expense of speed.
// Returns 0 if successful.

LIBYUV_API
int I420Scale(const uint8_t* src_y,
              int src_stride_y,
              const uint8_t* src_u,
              int src_stride_u,
              const uint8_t* src_v,
              int src_stride_v,
              int src_width,
              int src_height,
              uint8_t* dst_y,
              int dst_stride_y,
              uint8_t* dst_u,
              int dst_stride_u,
              uint8_t* dst_v,
              int dst_stride_v,
              int dst_width,
              int dst_height,
              enum FilterMode filtering);

LIBYUV_API
int I420Scale_16(const uint16_t* src_y,
                 int src_stride_y,
                 const uint16_t* src_u,
                 int src_stride_u,
                 const uint16_t* src_v,
                 int src_stride_v,
                 int src_width,
                 int src_height,
                 uint16_t* dst_y,
                 int dst_stride_y,
                 uint16_t* dst_u,
                 int dst_stride_u,
                 uint16_t* dst_v,
                 int dst_stride_v,
                 int dst_width,
                 int dst_height,
                 enum FilterMode filtering);

#ifdef __cplusplus
// Legacy API.  Deprecated.
LIBYUV_API
int Scale(const uint8_t* src_y,
          const uint8_t* src_u,
          const uint8_t* src_v,
          int src_stride_y,
          int src_stride_u,
          int src_stride_v,
          int src_width,
          int src_height,
          uint8_t* dst_y,
          uint8_t* dst_u,
          uint8_t* dst_v,
          int dst_stride_y,
          int dst_stride_u,
          int dst_stride_v,
          int dst_width,
          int dst_height,
          LIBYUV_BOOL interpolate);

// For testing, allow disabling of specialized scalers.
LIBYUV_API
void SetUseReferenceImpl(LIBYUV_BOOL use);
#endif  // __cplusplus

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif

#endif  // INCLUDE_LIBYUV_SCALE_H_
