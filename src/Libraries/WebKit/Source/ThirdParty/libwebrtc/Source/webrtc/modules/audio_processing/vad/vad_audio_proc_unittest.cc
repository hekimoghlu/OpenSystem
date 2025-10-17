/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 9, 2025.
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
// We don't test the value of pitch gain and lags as they are created by iSAC
// routines. However, interpolation of pitch-gain and lags is in a separate
// class and has its own unit-test.

#include "modules/audio_processing/vad/vad_audio_proc.h"

#include <math.h>
#include <stdio.h>

#include <string>

#include "modules/audio_processing/vad/common.h"
#include "test/gtest.h"
#include "test/testsupport/file_utils.h"

namespace webrtc {

TEST(AudioProcessingTest, DISABLED_ComputingFirstSpectralPeak) {
  VadAudioProc audioproc;

  std::string peak_file_name =
      test::ResourcePath("audio_processing/agc/agc_spectral_peak", "dat");
  FILE* peak_file = fopen(peak_file_name.c_str(), "rb");
  ASSERT_TRUE(peak_file != NULL);

  std::string pcm_file_name =
      test::ResourcePath("audio_processing/agc/agc_audio", "pcm");
  FILE* pcm_file = fopen(pcm_file_name.c_str(), "rb");
  ASSERT_TRUE(pcm_file != NULL);

  // Read 10 ms audio in each iteration.
  const size_t kDataLength = kLength10Ms;
  int16_t data[kDataLength] = {0};
  AudioFeatures features;
  double sp[kMaxNumFrames];
  while (fread(data, sizeof(int16_t), kDataLength, pcm_file) == kDataLength) {
    audioproc.ExtractFeatures(data, kDataLength, &features);
    if (features.num_frames > 0) {
      ASSERT_LT(features.num_frames, kMaxNumFrames);
      // Read reference values.
      const size_t num_frames = features.num_frames;
      ASSERT_EQ(num_frames, fread(sp, sizeof(sp[0]), num_frames, peak_file));
      for (size_t n = 0; n < features.num_frames; n++)
        EXPECT_NEAR(features.spectral_peak[n], sp[n], 3);
    }
  }

  fclose(peak_file);
  fclose(pcm_file);
}

}  // namespace webrtc
