/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 26, 2025.
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
#include "modules/audio_processing/vad/standalone_vad.h"

#include <string.h>

#include <memory>

#include "test/gtest.h"
#include "test/testsupport/file_utils.h"

namespace webrtc {

TEST(StandaloneVadTest, Api) {
  std::unique_ptr<StandaloneVad> vad(StandaloneVad::Create());
  int16_t data[kLength10Ms] = {0};

  // Valid frame length (for 32 kHz rate), but not what the VAD is expecting.
  EXPECT_EQ(-1, vad->AddAudio(data, 320));

  const size_t kMaxNumFrames = 3;
  double p[kMaxNumFrames];
  for (size_t n = 0; n < kMaxNumFrames; n++)
    EXPECT_EQ(0, vad->AddAudio(data, kLength10Ms));

  // Pretend `p` is shorter that it should be.
  EXPECT_EQ(-1, vad->GetActivity(p, kMaxNumFrames - 1));

  EXPECT_EQ(0, vad->GetActivity(p, kMaxNumFrames));

  // Ask for activity when buffer is empty.
  EXPECT_EQ(-1, vad->GetActivity(p, kMaxNumFrames));

  // Should reset and result in one buffer.
  for (size_t n = 0; n < kMaxNumFrames + 1; n++)
    EXPECT_EQ(0, vad->AddAudio(data, kLength10Ms));
  EXPECT_EQ(0, vad->GetActivity(p, 1));

  // Wrong modes
  EXPECT_EQ(-1, vad->set_mode(-1));
  EXPECT_EQ(-1, vad->set_mode(4));

  // Valid mode.
  const int kMode = 2;
  EXPECT_EQ(0, vad->set_mode(kMode));
  EXPECT_EQ(kMode, vad->mode());
}

#if defined(WEBRTC_IOS)
TEST(StandaloneVadTest, DISABLED_ActivityDetection) {
#else
TEST(StandaloneVadTest, ActivityDetection) {
#endif
  std::unique_ptr<StandaloneVad> vad(StandaloneVad::Create());
  const size_t kDataLength = kLength10Ms;
  int16_t data[kDataLength] = {0};

  FILE* pcm_file =
      fopen(test::ResourcePath("audio_processing/agc/agc_audio", "pcm").c_str(),
            "rb");
  ASSERT_TRUE(pcm_file != NULL);

  FILE* reference_file = fopen(
      test::ResourcePath("audio_processing/agc/agc_vad", "dat").c_str(), "rb");
  ASSERT_TRUE(reference_file != NULL);

  // Reference activities are prepared with 0 aggressiveness.
  ASSERT_EQ(0, vad->set_mode(0));

  // Stand-alone VAD can operate on 1, 2 or 3 frames of length 10 ms. The
  // reference file is created for 30 ms frame.
  const int kNumVadFramesToProcess = 3;
  int num_frames = 0;
  while (fread(data, sizeof(int16_t), kDataLength, pcm_file) == kDataLength) {
    vad->AddAudio(data, kDataLength);
    num_frames++;
    if (num_frames == kNumVadFramesToProcess) {
      num_frames = 0;
      int referece_activity;
      double p[kNumVadFramesToProcess];
      EXPECT_EQ(1u, fread(&referece_activity, sizeof(referece_activity), 1,
                          reference_file));
      int activity = vad->GetActivity(p, kNumVadFramesToProcess);
      EXPECT_EQ(referece_activity, activity);
      if (activity != 0) {
        // When active, probabilities are set to 0.5.
        for (int n = 0; n < kNumVadFramesToProcess; n++)
          EXPECT_EQ(0.5, p[n]);
      } else {
        // When inactive, probabilities are set to 0.01.
        for (int n = 0; n < kNumVadFramesToProcess; n++)
          EXPECT_EQ(0.01, p[n]);
      }
    }
  }
  fclose(reference_file);
  fclose(pcm_file);
}
}  // namespace webrtc
