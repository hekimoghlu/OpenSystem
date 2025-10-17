/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 14, 2024.
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
#include "common_audio/resampler/include/push_resampler.h"

#include "rtc_base/checks.h"  // RTC_DCHECK_IS_ON
#include "test/gtest.h"
#include "test/testsupport/rtc_expect_death.h"

// Quality testing of PushResampler is done in audio/remix_resample_unittest.cc.

namespace webrtc {

TEST(PushResamplerTest, VerifiesInputParameters) {
  PushResampler<int16_t> resampler1(160, 160, 1);
  PushResampler<int16_t> resampler2(160, 160, 2);
  PushResampler<int16_t> resampler3(160, 160, 8);
}

#if RTC_DCHECK_IS_ON && GTEST_HAS_DEATH_TEST && !defined(WEBRTC_ANDROID)
TEST(PushResamplerDeathTest, VerifiesBadInputParameters1) {
  RTC_EXPECT_DEATH(PushResampler<int16_t>(-1, 160, 1),
                   "src_samples_per_channel");
}

TEST(PushResamplerDeathTest, VerifiesBadInputParameters2) {
  RTC_EXPECT_DEATH(PushResampler<int16_t>(160, -1, 1),
                   "dst_samples_per_channel");
}

TEST(PushResamplerDeathTest, VerifiesBadInputParameters3) {
  RTC_EXPECT_DEATH(PushResampler<int16_t>(160, 16000, 0), "num_channels");
}
#endif

}  // namespace webrtc
