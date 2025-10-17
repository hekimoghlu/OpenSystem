/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 7, 2025.
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
#ifndef AOM_COMMON_RAWENC_H_
#define AOM_COMMON_RAWENC_H_

#include "aom/aom_decoder.h"
#include "common/md5_utils.h"
#include "common/tools_common.h"

#ifdef __cplusplus
extern "C" {
#endif

void raw_write_image_file(const aom_image_t *img, const int *planes,
                          const int num_planes, FILE *file);
void raw_update_image_md5(const aom_image_t *img, const int *planes,
                          const int num_planes, MD5Context *md5);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_COMMON_RAWENC_H_
