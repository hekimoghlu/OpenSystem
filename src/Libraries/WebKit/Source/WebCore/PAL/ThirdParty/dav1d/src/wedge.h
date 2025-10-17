/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 29, 2025.
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
#ifndef DAV1D_SRC_WEDGE_H
#define DAV1D_SRC_WEDGE_H

#include "src/levels.h"

void dav1d_init_wedge_masks(void);
extern const uint8_t *dav1d_wedge_masks[N_BS_SIZES][3 /* 444/luma, 422, 420 */]
                                 [2 /* sign */][16 /* wedge_idx */];

void dav1d_init_interintra_masks(void);
extern const uint8_t *dav1d_ii_masks[N_BS_SIZES][3 /* 444/luma, 422, 420 */]
                                    [N_INTER_INTRA_PRED_MODES];

#endif /* DAV1D_SRC_WEDGE_H */
