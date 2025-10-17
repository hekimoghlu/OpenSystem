/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 15, 2024.
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
#include "modules/audio_processing/agc2/limiter.h"

#include <algorithm>

#include "common_audio/include/audio_util.h"
#include "modules/audio_processing/agc2/agc2_common.h"
#include "modules/audio_processing/agc2/agc2_testing_common.h"
#include "modules/audio_processing/agc2/vector_float_frame.h"
#include "modules/audio_processing/logging/apm_data_dumper.h"
#include "rtc_base/gunit.h"

namespace webrtc {

TEST(Limiter, LimiterShouldConstructAndRun) {
  constexpr size_t kSamplesPerChannel = 480;
  ApmDataDumper apm_data_dumper(0);

  Limiter limiter(&apm_data_dumper, kSamplesPerChannel, "");

  std::array<float, kSamplesPerChannel> buffer;
  buffer.fill(kMaxAbsFloatS16Value);
  limiter.Process(
      DeinterleavedView<float>(buffer.data(), kSamplesPerChannel, 1));
}

TEST(Limiter, OutputVolumeAboveThreshold) {
  constexpr size_t kSamplesPerChannel = 480;
  const float input_level =
      (kMaxAbsFloatS16Value + DbfsToFloatS16(test::kLimiterMaxInputLevelDbFs)) /
      2.f;
  ApmDataDumper apm_data_dumper(0);

  Limiter limiter(&apm_data_dumper, kSamplesPerChannel, "");

  std::array<float, kSamplesPerChannel> buffer;

  // Give the level estimator time to adapt.
  for (int i = 0; i < 5; ++i) {
    std::fill(buffer.begin(), buffer.end(), input_level);
    limiter.Process(
        DeinterleavedView<float>(buffer.data(), kSamplesPerChannel, 1));
  }

  std::fill(buffer.begin(), buffer.end(), input_level);
  limiter.Process(
      DeinterleavedView<float>(buffer.data(), kSamplesPerChannel, 1));
  for (const auto& sample : buffer) {
    ASSERT_LT(0.9f * kMaxAbsFloatS16Value, sample);
  }
}

}  // namespace webrtc
