/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 16, 2024.
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
#ifndef AOM_AOM_DSP_ARM_REINTERPRET_NEON_H_
#define AOM_AOM_DSP_ARM_REINTERPRET_NEON_H_

#include <arm_neon.h>

#include "aom_dsp/aom_dsp_common.h"  // For AOM_FORCE_INLINE.
#include "config/aom_config.h"

#define REINTERPRET_NEON(u, to_sz, to_count, from_sz, from_count, n, q)     \
  static AOM_FORCE_INLINE u##int##to_sz##x##to_count##x##n##_t              \
      aom_reinterpret##q##_##u##to_sz##_##u##from_sz##_x##n(                \
          const u##int##from_sz##x##from_count##x##n##_t src) {             \
    u##int##to_sz##x##to_count##x##n##_t ret;                               \
    for (int i = 0; i < (n); ++i) {                                         \
      ret.val[i] = vreinterpret##q##_##u##to_sz##_##u##from_sz(src.val[i]); \
    }                                                                       \
    return ret;                                                             \
  }

REINTERPRET_NEON(u, 8, 8, 16, 4, 2, )    // uint8x8x2_t from uint16x4x2_t
REINTERPRET_NEON(u, 8, 16, 16, 8, 2, q)  // uint8x16x2_t from uint16x8x2_t

#endif  // AOM_AOM_DSP_ARM_REINTERPRET_NEON_H_
