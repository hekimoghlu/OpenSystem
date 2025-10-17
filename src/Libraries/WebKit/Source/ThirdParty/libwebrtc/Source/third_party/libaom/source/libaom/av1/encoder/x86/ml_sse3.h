/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 4, 2024.
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
#ifndef AOM_AV1_ENCODER_X86_ML_SSE3_H_
#define AOM_AV1_ENCODER_X86_ML_SSE3_H_

#include <pmmintrin.h>

void av1_nn_propagate_4to1_sse3(const float *const inputs,
                                const float *const weights,
                                __m128 *const output);

void av1_nn_propagate_4to4_sse3(const float *const inputs,
                                const float *const weights,
                                __m128 *const outputs, const int num_inputs);

void av1_nn_propagate_4to8_sse3(const float *const inputs,
                                const float *const weights, __m128 *const out_h,
                                __m128 *const out_l, const int num_inputs);

#endif  // AOM_AV1_ENCODER_X86_ML_SSE3_H_
