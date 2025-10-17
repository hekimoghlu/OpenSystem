/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 27, 2021.
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
#ifndef INCLUDE_LIBYUV_SCALE_RGB_H_
#define INCLUDE_LIBYUV_SCALE_RGB_H_

#include "libyuv/basic_types.h"
#include "libyuv/scale.h"  // For FilterMode

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

// RGB can be RAW, RGB24 or YUV24
// RGB scales 24 bit images by converting a row at a time to ARGB
// and using ARGB row functions to scale, then convert to RGB.
// TODO(fbarchard): Allow input/output formats to be specified.
LIBYUV_API
int RGBScale(const uint8_t* src_rgb,
             int src_stride_rgb,
             int src_width,
             int src_height,
             uint8_t* dst_rgb,
             int dst_stride_rgb,
             int dst_width,
             int dst_height,
             enum FilterMode filtering);

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif

#endif  // INCLUDE_LIBYUV_SCALE_UV_H_
