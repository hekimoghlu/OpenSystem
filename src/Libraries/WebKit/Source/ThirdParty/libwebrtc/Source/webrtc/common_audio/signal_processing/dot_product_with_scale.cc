/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 30, 2021.
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
#include "common_audio/signal_processing/dot_product_with_scale.h"

#include "rtc_base/numerics/safe_conversions.h"

int32_t WebRtcSpl_DotProductWithScale(const int16_t* vector1,
                                      const int16_t* vector2,
                                      size_t length,
                                      int scaling) {
  int64_t sum = 0;
  size_t i = 0;

  /* Unroll the loop to improve performance. */
  for (i = 0; i + 3 < length; i += 4) {
    sum += (vector1[i + 0] * vector2[i + 0]) >> scaling;
    sum += (vector1[i + 1] * vector2[i + 1]) >> scaling;
    sum += (vector1[i + 2] * vector2[i + 2]) >> scaling;
    sum += (vector1[i + 3] * vector2[i + 3]) >> scaling;
  }
  for (; i < length; i++) {
    sum += (vector1[i] * vector2[i]) >> scaling;
  }

  return rtc::saturated_cast<int32_t>(sum);
}
