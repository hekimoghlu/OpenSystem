/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 10, 2023.
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
#ifndef VPX_VP8_ENCODER_ETHREADING_H_
#define VPX_VP8_ENCODER_ETHREADING_H_

#include "vp8/encoder/onyx_int.h"

#ifdef __cplusplus
extern "C" {
#endif

struct VP8_COMP;
struct macroblock;

void vp8cx_init_mbrthread_data(struct VP8_COMP *cpi, struct macroblock *x,
                               MB_ROW_COMP *mbr_ei, int count);
int vp8cx_create_encoder_threads(struct VP8_COMP *cpi);
void vp8cx_remove_encoder_threads(struct VP8_COMP *cpi);

#ifdef __cplusplus
}
#endif

#endif  // VPX_VP8_ENCODER_ETHREADING_H_
