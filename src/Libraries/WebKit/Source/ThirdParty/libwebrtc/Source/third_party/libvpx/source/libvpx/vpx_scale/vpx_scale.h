/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 25, 2024.
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
#ifndef VPX_VPX_SCALE_VPX_SCALE_H_
#define VPX_VPX_SCALE_VPX_SCALE_H_

#include "vpx_scale/yv12config.h"

extern void vpx_scale_frame(YV12_BUFFER_CONFIG *src, YV12_BUFFER_CONFIG *dst,
                            unsigned char *temp_area, unsigned char temp_height,
                            unsigned int hscale, unsigned int hratio,
                            unsigned int vscale, unsigned int vratio,
                            unsigned int interlaced);

#endif  // VPX_VPX_SCALE_VPX_SCALE_H_
