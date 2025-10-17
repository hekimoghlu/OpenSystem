/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 4, 2024.
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
#ifndef VPX_Y4MENC_H_
#define VPX_Y4MENC_H_

#include "./tools_common.h"

#include "vpx/vpx_decoder.h"

#ifdef __cplusplus
extern "C" {
#endif

#define Y4M_BUFFER_SIZE 128

int y4m_write_file_header(char *buf, size_t len, int width, int height,
                          const struct VpxRational *framerate,
                          vpx_img_fmt_t fmt, unsigned int bit_depth);
int y4m_write_frame_header(char *buf, size_t len);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_Y4MENC_H_
