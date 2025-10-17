/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 11, 2024.
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
#ifndef AOM_COMMON_IVFENC_H_
#define AOM_COMMON_IVFENC_H_

#include "common/tools_common.h"

struct aom_codec_enc_cfg;
struct aom_codec_cx_pkt;

#ifdef __cplusplus
extern "C" {
#endif

void ivf_write_file_header(FILE *outfile, const struct aom_codec_enc_cfg *cfg,
                           uint32_t fourcc, int frame_cnt);

void ivf_write_frame_header(FILE *outfile, int64_t pts, size_t frame_size);

void ivf_write_frame_size(FILE *outfile, size_t frame_size);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // AOM_COMMON_IVFENC_H_
