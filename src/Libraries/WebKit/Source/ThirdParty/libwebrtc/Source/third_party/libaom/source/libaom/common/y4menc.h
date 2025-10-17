/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 24, 2021.
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
#ifndef AOM_COMMON_Y4MENC_H_
#define AOM_COMMON_Y4MENC_H_

#include "aom/aom_decoder.h"
#include "common/md5_utils.h"
#include "common/tools_common.h"

#ifdef __cplusplus
extern "C" {
#endif

#define Y4M_BUFFER_SIZE 256

int y4m_write_file_header(char *buf, size_t len, int width, int height,
                          const struct AvxRational *framerate, int monochrome,
                          aom_chroma_sample_position_t csp, aom_img_fmt_t fmt,
                          unsigned int bit_depth, aom_color_range_t range);
int y4m_write_frame_header(char *buf, size_t len);
void y4m_write_image_file(const aom_image_t *img, const int *planes,
                          FILE *file);
void y4m_update_image_md5(const aom_image_t *img, const int *planes,
                          MD5Context *md5);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_COMMON_Y4MENC_H_
