/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 12, 2023.
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
#ifndef VPX_VPX_DSP_QUANTIZE_H_
#define VPX_VPX_DSP_QUANTIZE_H_

#include "./vpx_config.h"
#include "vpx_dsp/vpx_dsp_common.h"

#ifdef __cplusplus
extern "C" {
#endif

void vpx_quantize_dc(const tran_low_t *coeff_ptr, int n_coeffs,
                     const int16_t *round_ptr, const int16_t quant,
                     tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr,
                     const int16_t dequant, uint16_t *eob_ptr);
void vpx_quantize_dc_32x32(const tran_low_t *coeff_ptr,
                           const int16_t *round_ptr, const int16_t quant,
                           tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr,
                           const int16_t dequant, uint16_t *eob_ptr);

#if CONFIG_VP9_HIGHBITDEPTH
void vpx_highbd_quantize_dc(const tran_low_t *coeff_ptr, int n_coeffs,
                            const int16_t *round_ptr, const int16_t quant,
                            tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr,
                            const int16_t dequant, uint16_t *eob_ptr);
void vpx_highbd_quantize_dc_32x32(const tran_low_t *coeff_ptr,
                                  const int16_t *round_ptr, const int16_t quant,
                                  tran_low_t *qcoeff_ptr,
                                  tran_low_t *dqcoeff_ptr,
                                  const int16_t dequant, uint16_t *eob_ptr);
#endif

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VPX_DSP_QUANTIZE_H_
