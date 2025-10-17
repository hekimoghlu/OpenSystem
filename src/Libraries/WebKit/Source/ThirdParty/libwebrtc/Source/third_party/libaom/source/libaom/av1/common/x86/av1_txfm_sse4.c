/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 26, 2023.
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
#include "config/av1_rtcd.h"

#include "av1/common/av1_txfm.h"
#include "av1/common/x86/av1_txfm_sse4.h"

// This function assumes `arr` is 16-byte aligned.
void av1_round_shift_array_sse4_1(int32_t *arr, int size, int bit) {
  __m128i *const vec = (__m128i *)arr;
  const int vec_size = size >> 2;
  av1_round_shift_array_32_sse4_1(vec, vec, vec_size, bit);
}
