/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 27, 2024.
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
#ifndef VPX_VP9_COMMON_VP9_SCAN_H_
#define VPX_VP9_COMMON_VP9_SCAN_H_

#include "vpx/vpx_integer.h"
#include "vpx_ports/mem.h"

#include "vp9/common/vp9_enums.h"
#include "vp9/common/vp9_blockd.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_NEIGHBORS 2

typedef struct ScanOrder {
  const int16_t *scan;
  const int16_t *iscan;
  const int16_t *neighbors;
} ScanOrder;

extern const ScanOrder vp9_default_scan_orders[TX_SIZES];
extern const ScanOrder vp9_scan_orders[TX_SIZES][TX_TYPES];

static INLINE int get_coef_context(const int16_t *neighbors,
                                   const uint8_t *token_cache, int c) {
  return (1 + token_cache[neighbors[MAX_NEIGHBORS * c + 0]] +
          token_cache[neighbors[MAX_NEIGHBORS * c + 1]]) >>
         1;
}

static INLINE const ScanOrder *get_scan(const MACROBLOCKD *xd, TX_SIZE tx_size,
                                        PLANE_TYPE type, int block_idx) {
  const MODE_INFO *const mi = xd->mi[0];

  if (is_inter_block(mi) || type != PLANE_TYPE_Y || xd->lossless) {
    return &vp9_default_scan_orders[tx_size];
  } else {
    const PREDICTION_MODE mode = get_y_mode(mi, block_idx);
    return &vp9_scan_orders[tx_size][intra_mode_to_tx_type_lookup[mode]];
  }
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP9_COMMON_VP9_SCAN_H_
