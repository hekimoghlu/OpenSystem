/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 3, 2025.
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
#ifndef VPX_VP9_COMMON_VP9_IDCT_H_
#define VPX_VP9_COMMON_VP9_IDCT_H_

#include <assert.h>

#include "./vpx_config.h"
#include "vp9/common/vp9_common.h"
#include "vp9/common/vp9_enums.h"
#include "vpx_dsp/inv_txfm.h"
#include "vpx_dsp/txfm_common.h"
#include "vpx_ports/mem.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*transform_1d)(const tran_low_t *, tran_low_t *);

typedef struct {
  transform_1d cols, rows;  // vertical and horizontal
} transform_2d;

#if CONFIG_VP9_HIGHBITDEPTH
typedef void (*highbd_transform_1d)(const tran_low_t *, tran_low_t *, int bd);

typedef struct {
  highbd_transform_1d cols, rows;  // vertical and horizontal
} highbd_transform_2d;
#endif  // CONFIG_VP9_HIGHBITDEPTH

void vp9_iwht4x4_add(const tran_low_t *input, uint8_t *dest, int stride,
                     int eob);
void vp9_idct4x4_add(const tran_low_t *input, uint8_t *dest, int stride,
                     int eob);
void vp9_idct8x8_add(const tran_low_t *input, uint8_t *dest, int stride,
                     int eob);
void vp9_idct16x16_add(const tran_low_t *input, uint8_t *dest, int stride,
                       int eob);
void vp9_idct32x32_add(const tran_low_t *input, uint8_t *dest, int stride,
                       int eob);

void vp9_iht4x4_add(TX_TYPE tx_type, const tran_low_t *input, uint8_t *dest,
                    int stride, int eob);
void vp9_iht8x8_add(TX_TYPE tx_type, const tran_low_t *input, uint8_t *dest,
                    int stride, int eob);
void vp9_iht16x16_add(TX_TYPE tx_type, const tran_low_t *input, uint8_t *dest,
                      int stride, int eob);

#if CONFIG_VP9_HIGHBITDEPTH
void vp9_highbd_iwht4x4_add(const tran_low_t *input, uint16_t *dest, int stride,
                            int eob, int bd);
void vp9_highbd_idct4x4_add(const tran_low_t *input, uint16_t *dest, int stride,
                            int eob, int bd);
void vp9_highbd_idct8x8_add(const tran_low_t *input, uint16_t *dest, int stride,
                            int eob, int bd);
void vp9_highbd_idct16x16_add(const tran_low_t *input, uint16_t *dest,
                              int stride, int eob, int bd);
void vp9_highbd_idct32x32_add(const tran_low_t *input, uint16_t *dest,
                              int stride, int eob, int bd);
void vp9_highbd_iht4x4_add(TX_TYPE tx_type, const tran_low_t *input,
                           uint16_t *dest, int stride, int eob, int bd);
void vp9_highbd_iht8x8_add(TX_TYPE tx_type, const tran_low_t *input,
                           uint16_t *dest, int stride, int eob, int bd);
void vp9_highbd_iht16x16_add(TX_TYPE tx_type, const tran_low_t *input,
                             uint16_t *dest, int stride, int eob, int bd);
#endif  // CONFIG_VP9_HIGHBITDEPTH
#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP9_COMMON_VP9_IDCT_H_
