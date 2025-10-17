/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 11, 2025.
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
#ifndef VPX_VP8_COMMON_RECONINTER_H_
#define VPX_VP8_COMMON_RECONINTER_H_

#ifdef __cplusplus
extern "C" {
#endif

void vp8_build_inter_predictors_mb(MACROBLOCKD *xd);
void vp8_build_inter16x16_predictors_mb(MACROBLOCKD *x, unsigned char *dst_y,
                                        unsigned char *dst_u,
                                        unsigned char *dst_v, int dst_ystride,
                                        int dst_uvstride);

void vp8_build_inter16x16_predictors_mby(MACROBLOCKD *x, unsigned char *dst_y,
                                         int dst_ystride);
void vp8_build_inter_predictors_b(BLOCKD *d, int pitch, unsigned char *base_pre,
                                  int pre_stride, vp8_subpix_fn_t sppf);

void vp8_build_inter16x16_predictors_mbuv(MACROBLOCKD *x);
void vp8_build_inter4x4_predictors_mbuv(MACROBLOCKD *x);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP8_COMMON_RECONINTER_H_
