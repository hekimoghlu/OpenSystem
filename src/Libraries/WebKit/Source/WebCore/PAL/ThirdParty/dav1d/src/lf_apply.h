/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 6, 2022.
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
#ifndef DAV1D_SRC_LF_APPLY_H
#define DAV1D_SRC_LF_APPLY_H

#include <stdint.h>

#include "common/bitdepth.h"

#include "src/internal.h"
#include "src/levels.h"

void bytefn(dav1d_loopfilter_sbrow_cols)(const Dav1dFrameContext *f,
                                         pixel *const p[3], Av1Filter *lflvl,
                                         int sby, int start_of_tile_row);
void bytefn(dav1d_loopfilter_sbrow_rows)(const Dav1dFrameContext *f,
                                         pixel *const p[3], Av1Filter *lflvl,
                                         int sby);

void bytefn(dav1d_copy_lpf)(Dav1dFrameContext *const f,
                            /*const*/ pixel *const src[3], int sby);

#endif /* DAV1D_SRC_LF_APPLY_H */
