/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 5, 2025.
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
#include <immintrin.h>

#include "config/aom_dsp_rtcd.h"

#include "aom_dsp/x86/common_avx2.h"
#include "aom_dsp/x86/lpf_common_sse2.h"
#include "aom/aom_integer.h"

void aom_highbd_lpf_horizontal_14_dual_avx2(
    uint16_t *s, int p, const uint8_t *blimit0, const uint8_t *limit0,
    const uint8_t *thresh0, const uint8_t *blimit1, const uint8_t *limit1,
    const uint8_t *thresh1, int bd) {
  aom_highbd_lpf_horizontal_14_dual_sse2(s, p, blimit0, limit0, thresh0,
                                         blimit1, limit1, thresh1, bd);
}

void aom_highbd_lpf_vertical_14_dual_avx2(
    uint16_t *s, int p, const uint8_t *blimit0, const uint8_t *limit0,
    const uint8_t *thresh0, const uint8_t *blimit1, const uint8_t *limit1,
    const uint8_t *thresh1, int bd) {
  aom_highbd_lpf_vertical_14_dual_sse2(s, p, blimit0, limit0, thresh0, blimit1,
                                       limit1, thresh1, bd);
}

void aom_highbd_lpf_horizontal_4_dual_avx2(
    uint16_t *s, int p, const uint8_t *blimit0, const uint8_t *limit0,
    const uint8_t *thresh0, const uint8_t *blimit1, const uint8_t *limit1,
    const uint8_t *thresh1, int bd) {
  aom_highbd_lpf_horizontal_4_dual_sse2(s, p, blimit0, limit0, thresh0, blimit1,
                                        limit1, thresh1, bd);
}

void aom_highbd_lpf_horizontal_8_dual_avx2(
    uint16_t *s, int p, const uint8_t *blimit0, const uint8_t *limit0,
    const uint8_t *thresh0, const uint8_t *blimit1, const uint8_t *limit1,
    const uint8_t *thresh1, int bd) {
  aom_highbd_lpf_horizontal_8_dual_sse2(s, p, blimit0, limit0, thresh0, blimit1,
                                        limit1, thresh1, bd);
}

void aom_highbd_lpf_vertical_4_dual_avx2(
    uint16_t *s, int p, const uint8_t *blimit0, const uint8_t *limit0,
    const uint8_t *thresh0, const uint8_t *blimit1, const uint8_t *limit1,
    const uint8_t *thresh1, int bd) {
  aom_highbd_lpf_vertical_4_dual_sse2(s, p, blimit0, limit0, thresh0, blimit1,
                                      limit1, thresh1, bd);
}

void aom_highbd_lpf_vertical_8_dual_avx2(
    uint16_t *s, int p, const uint8_t *blimit0, const uint8_t *limit0,
    const uint8_t *thresh0, const uint8_t *blimit1, const uint8_t *limit1,
    const uint8_t *thresh1, int bd) {
  aom_highbd_lpf_vertical_8_dual_sse2(s, p, blimit0, limit0, thresh0, blimit1,
                                      limit1, thresh1, bd);
}
