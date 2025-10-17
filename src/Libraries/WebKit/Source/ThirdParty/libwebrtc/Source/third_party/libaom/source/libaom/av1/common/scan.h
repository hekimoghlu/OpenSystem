/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 16, 2024.
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
#ifndef AOM_AV1_COMMON_SCAN_H_
#define AOM_AV1_COMMON_SCAN_H_

#include "aom/aom_integer.h"
#include "aom_ports/mem.h"

#include "av1/common/av1_common_int.h"
#include "av1/common/blockd.h"
#include "av1/common/enums.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_NEIGHBORS 2

enum {
  SCAN_MODE_ZIG_ZAG,
  SCAN_MODE_COL_DIAG,
  SCAN_MODE_ROW_DIAG,
  SCAN_MODE_COL_1D,
  SCAN_MODE_ROW_1D,
  SCAN_MODES
} UENUM1BYTE(SCAN_MODE);

extern const SCAN_ORDER av1_scan_orders[TX_SIZES_ALL][TX_TYPES];

void av1_deliver_eob_threshold(const AV1_COMMON *cm, MACROBLOCKD *xd);

static inline const SCAN_ORDER *get_default_scan(TX_SIZE tx_size,
                                                 TX_TYPE tx_type) {
  return &av1_scan_orders[tx_size][tx_type];
}

static inline const SCAN_ORDER *get_scan(TX_SIZE tx_size, TX_TYPE tx_type) {
  return get_default_scan(tx_size, tx_type);
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_COMMON_SCAN_H_
