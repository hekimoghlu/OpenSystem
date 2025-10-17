/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 10, 2025.
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
#ifndef MODULES_AUDIO_PROCESSING_AEC3_ADAPTIVE_FIR_FILTER_ERL_H_
#define MODULES_AUDIO_PROCESSING_AEC3_ADAPTIVE_FIR_FILTER_ERL_H_

#include <stddef.h>

#include <array>
#include <vector>

#include "api/array_view.h"
#include "modules/audio_processing/aec3/aec3_common.h"
#include "rtc_base/system/arch.h"

namespace webrtc {
namespace aec3 {

// Computes and stores the echo return loss estimate of the filter, which is the
// sum of the partition frequency responses.
void ErlComputer(const std::vector<std::array<float, kFftLengthBy2Plus1>>& H2,
                 rtc::ArrayView<float> erl);
#if defined(WEBRTC_HAS_NEON)
void ErlComputer_NEON(
    const std::vector<std::array<float, kFftLengthBy2Plus1>>& H2,
    rtc::ArrayView<float> erl);
#endif
#if defined(WEBRTC_ARCH_X86_FAMILY)
void ErlComputer_SSE2(
    const std::vector<std::array<float, kFftLengthBy2Plus1>>& H2,
    rtc::ArrayView<float> erl);

void ErlComputer_AVX2(
    const std::vector<std::array<float, kFftLengthBy2Plus1>>& H2,
    rtc::ArrayView<float> erl);
#endif

}  // namespace aec3

// Computes the echo return loss based on a frequency response.
void ComputeErl(const Aec3Optimization& optimization,
                const std::vector<std::array<float, kFftLengthBy2Plus1>>& H2,
                rtc::ArrayView<float> erl);

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AEC3_ADAPTIVE_FIR_FILTER_ERL_H_
