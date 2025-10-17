/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 20, 2022.
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
#ifndef COMMON_AUDIO_VAD_VAD_UNITTEST_H_
#define COMMON_AUDIO_VAD_VAD_UNITTEST_H_

#include <stddef.h>  // size_t

#include "test/gtest.h"

namespace webrtc {
namespace test {

// Modes we support
const int kModes[] = {0, 1, 2, 3};
const size_t kModesSize = sizeof(kModes) / sizeof(*kModes);

// Rates we support.
const int kRates[] = {8000, 12000, 16000, 24000, 32000, 48000};
const size_t kRatesSize = sizeof(kRates) / sizeof(*kRates);

// Frame lengths we support.
const size_t kMaxFrameLength = 1440;
const size_t kFrameLengths[] = {
    80, 120, 160, 240, 320, 480, 640, 960, kMaxFrameLength};
const size_t kFrameLengthsSize = sizeof(kFrameLengths) / sizeof(*kFrameLengths);

}  // namespace test
}  // namespace webrtc

class VadTest : public ::testing::Test {
 protected:
  VadTest();
  void SetUp() override;
  void TearDown() override;

  // Returns true if the rate and frame length combination is valid.
  bool ValidRatesAndFrameLengths(int rate, size_t frame_length);
};

#endif  // COMMON_AUDIO_VAD_VAD_UNITTEST_H_
