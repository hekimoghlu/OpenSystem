/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 29, 2024.
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
#ifndef VPX_VP9_COMMON_VP9_RECONINTRA_H_
#define VPX_VP9_COMMON_VP9_RECONINTRA_H_

#include "vpx/vpx_integer.h"
#include "vp9/common/vp9_blockd.h"

#ifdef __cplusplus
extern "C" {
#endif

void vp9_init_intra_predictors(void);

void vp9_predict_intra_block(const MACROBLOCKD *xd, int bwl_in, TX_SIZE tx_size,
                             PREDICTION_MODE mode, const uint8_t *ref,
                             int ref_stride, uint8_t *dst, int dst_stride,
                             int aoff, int loff, int plane);
#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP9_COMMON_VP9_RECONINTRA_H_
