/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 13, 2024.
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
#include <stdlib.h>

#include "common_audio/vad/vad_unittest.h"
#include "test/gtest.h"

extern "C" {
#include "common_audio/vad/vad_core.h"
#include "common_audio/vad/vad_filterbank.h"
}

namespace webrtc {
namespace test {

const int kNumValidFrameLengths = 3;

TEST_F(VadTest, vad_filterbank) {
  VadInstT* self = reinterpret_cast<VadInstT*>(malloc(sizeof(VadInstT)));
  static const int16_t kReference[kNumValidFrameLengths] = {48, 11, 11};
  static const int16_t kFeatures[kNumValidFrameLengths * kNumChannels] = {
      1213, 759,  587,  462,  434,  272,  1479, 1385, 1291,
      1200, 1103, 1099, 1732, 1692, 1681, 1629, 1436, 1436};
  static const int16_t kOffsetVector[kNumChannels] = {368, 368, 272,
                                                      176, 176, 176};
  int16_t features[kNumChannels];

  // Construct a speech signal that will trigger the VAD in all modes. It is
  // known that (i * i) will wrap around, but that doesn't matter in this case.
  int16_t speech[kMaxFrameLength];
  for (size_t i = 0; i < kMaxFrameLength; ++i) {
    speech[i] = static_cast<int16_t>(i * i);
  }

  int frame_length_index = 0;
  ASSERT_EQ(0, WebRtcVad_InitCore(self));
  for (size_t j = 0; j < kFrameLengthsSize; ++j) {
    if (ValidRatesAndFrameLengths(8000, kFrameLengths[j])) {
      EXPECT_EQ(kReference[frame_length_index],
                WebRtcVad_CalculateFeatures(self, speech, kFrameLengths[j],
                                            features));
      for (int k = 0; k < kNumChannels; ++k) {
        EXPECT_EQ(kFeatures[k + frame_length_index * kNumChannels],
                  features[k]);
      }
      frame_length_index++;
    }
  }
  EXPECT_EQ(kNumValidFrameLengths, frame_length_index);

  // Verify that all zeros in gives kOffsetVector out.
  memset(speech, 0, sizeof(speech));
  ASSERT_EQ(0, WebRtcVad_InitCore(self));
  for (size_t j = 0; j < kFrameLengthsSize; ++j) {
    if (ValidRatesAndFrameLengths(8000, kFrameLengths[j])) {
      EXPECT_EQ(0, WebRtcVad_CalculateFeatures(self, speech, kFrameLengths[j],
                                               features));
      for (int k = 0; k < kNumChannels; ++k) {
        EXPECT_EQ(kOffsetVector[k], features[k]);
      }
    }
  }

  // Verify that all ones in gives kOffsetVector out. Any other constant input
  // will have a small impact in the sub bands.
  for (size_t i = 0; i < kMaxFrameLength; ++i) {
    speech[i] = 1;
  }
  for (size_t j = 0; j < kFrameLengthsSize; ++j) {
    if (ValidRatesAndFrameLengths(8000, kFrameLengths[j])) {
      ASSERT_EQ(0, WebRtcVad_InitCore(self));
      EXPECT_EQ(0, WebRtcVad_CalculateFeatures(self, speech, kFrameLengths[j],
                                               features));
      for (int k = 0; k < kNumChannels; ++k) {
        EXPECT_EQ(kOffsetVector[k], features[k]);
      }
    }
  }

  free(self);
}
}  // namespace test
}  // namespace webrtc
