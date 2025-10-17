/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 3, 2025.
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
#ifndef AOM_AV1_COMMON_IDCT_H_
#define AOM_AV1_COMMON_IDCT_H_

#include "config/aom_config.h"

#include "av1/common/blockd.h"
#include "av1/common/common.h"
#include "av1/common/enums.h"
#include "aom_dsp/txfm_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*transform_1d)(const tran_low_t *, tran_low_t *);

typedef struct {
  transform_1d cols, rows;  // vertical and horizontal
} transform_2d;

#define MAX_TX_SCALE 1
int av1_get_tx_scale(const TX_SIZE tx_size);

void av1_inverse_transform_block(const MACROBLOCKD *xd,
                                 const tran_low_t *dqcoeff, int plane,
                                 TX_TYPE tx_type, TX_SIZE tx_size, uint8_t *dst,
                                 int stride, int eob, int reduced_tx_set);
void av1_highbd_iwht4x4_add(const tran_low_t *input, uint8_t *dest, int stride,
                            int eob, int bd);

static inline const int32_t *cast_to_int32(const tran_low_t *input) {
  assert(sizeof(int32_t) == sizeof(tran_low_t));
  return (const int32_t *)input;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_COMMON_IDCT_H_
