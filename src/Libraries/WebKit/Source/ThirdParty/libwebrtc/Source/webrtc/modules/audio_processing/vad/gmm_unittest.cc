/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 22, 2022.
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
#include "modules/audio_processing/vad/gmm.h"

#include <math.h>

#include "modules/audio_processing/vad/noise_gmm_tables.h"
#include "modules/audio_processing/vad/voice_gmm_tables.h"
#include "test/gtest.h"

namespace webrtc {

TEST(GmmTest, EvaluateGmm) {
  GmmParameters noise_gmm;
  GmmParameters voice_gmm;

  // Setup noise GMM.
  noise_gmm.dimension = kNoiseGmmDim;
  noise_gmm.num_mixtures = kNoiseGmmNumMixtures;
  noise_gmm.weight = kNoiseGmmWeights;
  noise_gmm.mean = &kNoiseGmmMean[0][0];
  noise_gmm.covar_inverse = &kNoiseGmmCovarInverse[0][0][0];

  // Setup voice GMM.
  voice_gmm.dimension = kVoiceGmmDim;
  voice_gmm.num_mixtures = kVoiceGmmNumMixtures;
  voice_gmm.weight = kVoiceGmmWeights;
  voice_gmm.mean = &kVoiceGmmMean[0][0];
  voice_gmm.covar_inverse = &kVoiceGmmCovarInverse[0][0][0];

  // Test vectors. These are the mean of the GMM means.
  const double kXVoice[kVoiceGmmDim] = {-1.35893162459863, 602.862491970368,
                                        178.022069191324};
  const double kXNoise[kNoiseGmmDim] = {-2.33443722724409, 2827.97828765184,
                                        141.114178166812};

  // Expected pdf values. These values are computed in MATLAB using EvalGmm.m
  const double kPdfNoise = 1.88904409403101e-07;
  const double kPdfVoice = 1.30453996982266e-06;

  // Relative error should be smaller that the following value.
  const double kAcceptedRelativeErr = 1e-10;

  // Test Voice.
  double pdf = EvaluateGmm(kXVoice, voice_gmm);
  EXPECT_GT(pdf, 0);
  double relative_error = fabs(pdf - kPdfVoice) / kPdfVoice;
  EXPECT_LE(relative_error, kAcceptedRelativeErr);

  // Test Noise.
  pdf = EvaluateGmm(kXNoise, noise_gmm);
  EXPECT_GT(pdf, 0);
  relative_error = fabs(pdf - kPdfNoise) / kPdfNoise;
  EXPECT_LE(relative_error, kAcceptedRelativeErr);
}

}  // namespace webrtc
