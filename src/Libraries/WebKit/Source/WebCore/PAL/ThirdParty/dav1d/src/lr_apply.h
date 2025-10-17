/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 17, 2022.
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
#ifndef DAV1D_SRC_LR_APPLY_H
#define DAV1D_SRC_LR_APPLY_H

#include <stdint.h>
#include <stddef.h>

#include "common/bitdepth.h"

#include "src/internal.h"

enum LrRestorePlanes {
    LR_RESTORE_Y = 1 << 0,
    LR_RESTORE_U = 1 << 1,
    LR_RESTORE_V = 1 << 2,
};

void bytefn(dav1d_lr_sbrow)(Dav1dFrameContext *const f, pixel *const dst[3],
                            int sby);

#endif /* DAV1D_SRC_LR_APPLY_H */
