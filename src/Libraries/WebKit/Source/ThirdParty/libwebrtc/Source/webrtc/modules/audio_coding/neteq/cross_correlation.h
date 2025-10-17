/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 3, 2024.
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
#ifndef MODULES_AUDIO_CODING_NETEQ_CROSS_CORRELATION_H_
#define MODULES_AUDIO_CODING_NETEQ_CROSS_CORRELATION_H_

#include <stddef.h>
#include <stdint.h>

namespace webrtc {

// The function calculates the cross-correlation between two sequences
// `sequence_1` and `sequence_2`. `sequence_1` is taken as reference, with
// `sequence_1_length` as its length. `sequence_2` slides for the calculation of
// cross-correlation. The result will be saved in `cross_correlation`.
// `cross_correlation_length` correlation points are calculated.
// The corresponding lag starts from 0, and increases with a step of
// `cross_correlation_step`. The result is without normalization. To avoid
// overflow, the result will be right shifted. The amount of shifts will be
// returned.
//
// Input:
//     - sequence_1     : First sequence (reference).
//     - sequence_2     : Second sequence (sliding during calculation).
//     - sequence_1_length : Length of `sequence_1`.
//     - cross_correlation_length : Number of cross-correlations to calculate.
//     - cross_correlation_step : Step in the lag for the cross-correlation.
//
// Output:
//      - cross_correlation : The cross-correlation in Q(-right_shifts)
//
// Return:
//      Number of right shifts in cross_correlation.

int CrossCorrelationWithAutoShift(const int16_t* sequence_1,
                                  const int16_t* sequence_2,
                                  size_t sequence_1_length,
                                  size_t cross_correlation_length,
                                  int cross_correlation_step,
                                  int32_t* cross_correlation);

}  // namespace webrtc

#endif  // MODULES_AUDIO_CODING_NETEQ_CROSS_CORRELATION_H_
