/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 20, 2022.
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
#ifndef VPX_VP9_DECODER_VP9_DECODEMV_H_
#define VPX_VP9_DECODER_VP9_DECODEMV_H_

#include "vpx_dsp/bitreader.h"

#include "vp9/decoder/vp9_decoder.h"

#ifdef __cplusplus
extern "C" {
#endif

void vp9_read_mode_info(TileWorkerData *twd, VP9Decoder *const pbi, int mi_row,
                        int mi_col, int x_mis, int y_mis);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP9_DECODER_VP9_DECODEMV_H_
