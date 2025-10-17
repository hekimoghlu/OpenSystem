/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 17, 2025.
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
#ifndef VPX_VP8_COMMON_EXTEND_H_
#define VPX_VP8_COMMON_EXTEND_H_

#include "vpx_scale/yv12config.h"

#ifdef __cplusplus
extern "C" {
#endif

void vp8_extend_mb_row(YV12_BUFFER_CONFIG *ybf, unsigned char *YPtr,
                       unsigned char *UPtr, unsigned char *VPtr);
void vp8_copy_and_extend_frame(YV12_BUFFER_CONFIG *src,
                               YV12_BUFFER_CONFIG *dst);
void vp8_copy_and_extend_frame_with_rect(YV12_BUFFER_CONFIG *src,
                                         YV12_BUFFER_CONFIG *dst, int srcy,
                                         int srcx, int srch, int srcw);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP8_COMMON_EXTEND_H_
