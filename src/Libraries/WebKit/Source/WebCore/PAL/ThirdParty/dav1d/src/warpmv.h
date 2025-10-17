/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 13, 2025.
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
#ifndef DAV1D_SRC_WARPMV_H
#define DAV1D_SRC_WARPMV_H

#include "src/levels.h"

int dav1d_get_shear_params(Dav1dWarpedMotionParams *wm);
int dav1d_find_affine_int(const int (*pts)[2][2], int np, int bw4, int bh4,
                          mv mv, Dav1dWarpedMotionParams *wm, int bx, int by);
void dav1d_set_affine_mv2d(int bw4, int bh4,
                           mv mv, Dav1dWarpedMotionParams *wm, int bx, int by);

#endif /* DAV1D_SRC_WARPMV_H */
