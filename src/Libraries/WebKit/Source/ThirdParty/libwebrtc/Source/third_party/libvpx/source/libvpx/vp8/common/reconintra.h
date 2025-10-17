/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 24, 2024.
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
#ifndef VPX_VP8_COMMON_RECONINTRA_H_
#define VPX_VP8_COMMON_RECONINTRA_H_

#include "vp8/common/blockd.h"

#ifdef __cplusplus
extern "C" {
#endif

void vp8_build_intra_predictors_mby_s(MACROBLOCKD *x, unsigned char *yabove_row,
                                      unsigned char *yleft, int left_stride,
                                      unsigned char *ypred_ptr, int y_stride);

void vp8_build_intra_predictors_mbuv_s(
    MACROBLOCKD *x, unsigned char *uabove_row, unsigned char *vabove_row,
    unsigned char *uleft, unsigned char *vleft, int left_stride,
    unsigned char *upred_ptr, unsigned char *vpred_ptr, int pred_stride);

void vp8_init_intra_predictors(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP8_COMMON_RECONINTRA_H_
