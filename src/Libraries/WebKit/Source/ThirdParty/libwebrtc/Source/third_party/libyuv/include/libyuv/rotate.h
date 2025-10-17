/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 15, 2022.
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
#ifndef INCLUDE_LIBYUV_ROTATE_H_
#define INCLUDE_LIBYUV_ROTATE_H_

#include "libyuv/basic_types.h"

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

// Supported rotation.
typedef enum RotationMode {
  kRotate0 = 0,      // No rotation.
  kRotate90 = 90,    // Rotate 90 degrees clockwise.
  kRotate180 = 180,  // Rotate 180 degrees.
  kRotate270 = 270,  // Rotate 270 degrees clockwise.

  // Deprecated.
  kRotateNone = 0,
  kRotateClockwise = 90,
  kRotateCounterClockwise = 270,
} RotationModeEnum;

// Rotate I420 frame.
LIBYUV_API
int I420Rotate(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int width,
               int height,
               enum RotationMode mode);

// Rotate I422 frame.
LIBYUV_API
int I422Rotate(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int width,
               int height,
               enum RotationMode mode);

// Rotate I444 frame.
LIBYUV_API
int I444Rotate(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int width,
               int height,
               enum RotationMode mode);

// Rotate I010 frame.
LIBYUV_API
int I010Rotate(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint16_t* dst_y,
               int dst_stride_y,
               uint16_t* dst_u,
               int dst_stride_u,
               uint16_t* dst_v,
               int dst_stride_v,
               int width,
               int height,
               enum RotationMode mode);

// Rotate I210 frame.
LIBYUV_API
int I210Rotate(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint16_t* dst_y,
               int dst_stride_y,
               uint16_t* dst_u,
               int dst_stride_u,
               uint16_t* dst_v,
               int dst_stride_v,
               int width,
               int height,
               enum RotationMode mode);

// Rotate I410 frame.
LIBYUV_API
int I410Rotate(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint16_t* dst_y,
               int dst_stride_y,
               uint16_t* dst_u,
               int dst_stride_u,
               uint16_t* dst_v,
               int dst_stride_v,
               int width,
               int height,
               enum RotationMode mode);

// Rotate NV12 input and store in I420.
LIBYUV_API
int NV12ToI420Rotate(const uint8_t* src_y,
                     int src_stride_y,
                     const uint8_t* src_uv,
                     int src_stride_uv,
                     uint8_t* dst_y,
                     int dst_stride_y,
                     uint8_t* dst_u,
                     int dst_stride_u,
                     uint8_t* dst_v,
                     int dst_stride_v,
                     int width,
                     int height,
                     enum RotationMode mode);

// Convert Android420 to I420 with rotation.
// "rotation" can be 0, 90, 180 or 270.
LIBYUV_API
int Android420ToI420Rotate(const uint8_t* src_y,
                           int src_stride_y,
                           const uint8_t* src_u,
                           int src_stride_u,
                           const uint8_t* src_v,
                           int src_stride_v,
                           int src_pixel_stride_uv,
                           uint8_t* dst_y,
                           int dst_stride_y,
                           uint8_t* dst_u,
                           int dst_stride_u,
                           uint8_t* dst_v,
                           int dst_stride_v,
                           int width,
                           int height,
                           enum RotationMode rotation);

// Rotate a plane by 0, 90, 180, or 270.
LIBYUV_API
int RotatePlane(const uint8_t* src,
                int src_stride,
                uint8_t* dst,
                int dst_stride,
                int width,
                int height,
                enum RotationMode mode);

// Rotate planes by 90, 180, 270. Deprecated.
LIBYUV_API
void RotatePlane90(const uint8_t* src,
                   int src_stride,
                   uint8_t* dst,
                   int dst_stride,
                   int width,
                   int height);

LIBYUV_API
void RotatePlane180(const uint8_t* src,
                    int src_stride,
                    uint8_t* dst,
                    int dst_stride,
                    int width,
                    int height);

LIBYUV_API
void RotatePlane270(const uint8_t* src,
                    int src_stride,
                    uint8_t* dst,
                    int dst_stride,
                    int width,
                    int height);

// Rotate a plane by 0, 90, 180, or 270.
LIBYUV_API
int RotatePlane_16(const uint16_t* src,
                   int src_stride,
                   uint16_t* dst,
                   int dst_stride,
                   int width,
                   int height,
                   enum RotationMode mode);

// Rotations for when U and V are interleaved.
// These functions take one UV input pointer and
// split the data into two buffers while
// rotating them.
// width and height expected to be half size for NV12.
LIBYUV_API
int SplitRotateUV(const uint8_t* src_uv,
                  int src_stride_uv,
                  uint8_t* dst_u,
                  int dst_stride_u,
                  uint8_t* dst_v,
                  int dst_stride_v,
                  int width,
                  int height,
                  enum RotationMode mode);

LIBYUV_API
void SplitRotateUV90(const uint8_t* src,
                     int src_stride,
                     uint8_t* dst_a,
                     int dst_stride_a,
                     uint8_t* dst_b,
                     int dst_stride_b,
                     int width,
                     int height);

LIBYUV_API
void SplitRotateUV180(const uint8_t* src,
                      int src_stride,
                      uint8_t* dst_a,
                      int dst_stride_a,
                      uint8_t* dst_b,
                      int dst_stride_b,
                      int width,
                      int height);

LIBYUV_API
void SplitRotateUV270(const uint8_t* src,
                      int src_stride,
                      uint8_t* dst_a,
                      int dst_stride_a,
                      uint8_t* dst_b,
                      int dst_stride_b,
                      int width,
                      int height);

// The 90 and 270 functions are based on transposes.
// Doing a transpose with reversing the read/write
// order will result in a rotation by +- 90 degrees.
// Deprecated.
LIBYUV_API
void TransposePlane(const uint8_t* src,
                    int src_stride,
                    uint8_t* dst,
                    int dst_stride,
                    int width,
                    int height);

LIBYUV_API
void SplitTransposeUV(const uint8_t* src,
                      int src_stride,
                      uint8_t* dst_a,
                      int dst_stride_a,
                      uint8_t* dst_b,
                      int dst_stride_b,
                      int width,
                      int height);

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif

#endif  // INCLUDE_LIBYUV_ROTATE_H_
