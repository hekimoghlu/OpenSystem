/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 22, 2025.
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
// Unit tests for test InputAudioFile class.

#include "modules/audio_coding/neteq/tools/input_audio_file.h"

#include "rtc_base/numerics/safe_conversions.h"
#include "test/gtest.h"

namespace webrtc {
namespace test {

TEST(TestInputAudioFile, DuplicateInterleaveSeparateSrcDst) {
  static const size_t kSamples = 10;
  static const size_t kChannels = 2;
  int16_t input[kSamples];
  for (size_t i = 0; i < kSamples; ++i) {
    input[i] = rtc::checked_cast<int16_t>(i);
  }
  int16_t output[kSamples * kChannels];
  InputAudioFile::DuplicateInterleaved(input, kSamples, kChannels, output);

  // Verify output
  int16_t* output_ptr = output;
  for (size_t i = 0; i < kSamples; ++i) {
    for (size_t j = 0; j < kChannels; ++j) {
      EXPECT_EQ(static_cast<int16_t>(i), *output_ptr++);
    }
  }
}

TEST(TestInputAudioFile, DuplicateInterleaveSameSrcDst) {
  static const size_t kSamples = 10;
  static const size_t kChannels = 5;
  int16_t input[kSamples * kChannels];
  for (size_t i = 0; i < kSamples; ++i) {
    input[i] = rtc::checked_cast<int16_t>(i);
  }
  InputAudioFile::DuplicateInterleaved(input, kSamples, kChannels, input);

  // Verify output
  int16_t* output_ptr = input;
  for (size_t i = 0; i < kSamples; ++i) {
    for (size_t j = 0; j < kChannels; ++j) {
      EXPECT_EQ(static_cast<int16_t>(i), *output_ptr++);
    }
  }
}

}  // namespace test
}  // namespace webrtc
