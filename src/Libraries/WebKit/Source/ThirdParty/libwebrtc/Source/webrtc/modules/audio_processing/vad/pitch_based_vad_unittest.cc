/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 2, 2022.
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
#include "modules/audio_processing/vad/pitch_based_vad.h"

#include <math.h>
#include <stdio.h>

#include <string>

#include "test/gtest.h"
#include "test/testsupport/file_utils.h"

namespace webrtc {

TEST(PitchBasedVadTest, VoicingProbabilityTest) {
  std::string spectral_peak_file_name =
      test::ResourcePath("audio_processing/agc/agc_spectral_peak", "dat");
  FILE* spectral_peak_file = fopen(spectral_peak_file_name.c_str(), "rb");
  ASSERT_TRUE(spectral_peak_file != NULL);

  std::string pitch_gain_file_name =
      test::ResourcePath("audio_processing/agc/agc_pitch_gain", "dat");
  FILE* pitch_gain_file = fopen(pitch_gain_file_name.c_str(), "rb");
  ASSERT_TRUE(pitch_gain_file != NULL);

  std::string pitch_lag_file_name =
      test::ResourcePath("audio_processing/agc/agc_pitch_lag", "dat");
  FILE* pitch_lag_file = fopen(pitch_lag_file_name.c_str(), "rb");
  ASSERT_TRUE(pitch_lag_file != NULL);

  std::string voicing_prob_file_name =
      test::ResourcePath("audio_processing/agc/agc_voicing_prob", "dat");
  FILE* voicing_prob_file = fopen(voicing_prob_file_name.c_str(), "rb");
  ASSERT_TRUE(voicing_prob_file != NULL);

  PitchBasedVad vad_;

  double reference_activity_probability;

  AudioFeatures audio_features;
  memset(&audio_features, 0, sizeof(audio_features));
  audio_features.num_frames = 1;
  while (fread(audio_features.spectral_peak,
               sizeof(audio_features.spectral_peak[0]), 1,
               spectral_peak_file) == 1u) {
    double p;
    ASSERT_EQ(1u, fread(audio_features.log_pitch_gain,
                        sizeof(audio_features.log_pitch_gain[0]), 1,
                        pitch_gain_file));
    ASSERT_EQ(1u,
              fread(audio_features.pitch_lag_hz,
                    sizeof(audio_features.pitch_lag_hz[0]), 1, pitch_lag_file));
    ASSERT_EQ(1u, fread(&reference_activity_probability,
                        sizeof(reference_activity_probability), 1,
                        voicing_prob_file));

    p = 0.5;  // Initialize to the neutral value for combining probabilities.
    EXPECT_EQ(0, vad_.VoicingProbability(audio_features, &p));
    EXPECT_NEAR(p, reference_activity_probability, 0.01);
  }

  fclose(spectral_peak_file);
  fclose(pitch_gain_file);
  fclose(pitch_lag_file);
}

}  // namespace webrtc
