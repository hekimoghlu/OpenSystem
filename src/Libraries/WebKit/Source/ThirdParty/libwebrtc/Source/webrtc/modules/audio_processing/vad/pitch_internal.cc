/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 17, 2023.
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
#include "modules/audio_processing/vad/pitch_internal.h"

#include <cmath>

namespace webrtc {

// A 4-to-3 linear interpolation.
// The interpolation constants are derived as following:
// Input pitch parameters are updated every 7.5 ms. Within a 30-ms interval
// we are interested in pitch parameters of 0-5 ms, 10-15ms and 20-25ms. This is
// like interpolating 4-to-6 and keep the odd samples.
// The reason behind this is that LPC coefficients are computed for the first
// half of each 10ms interval.
static void PitchInterpolation(double old_val, const double* in, double* out) {
  out[0] = 1. / 6. * old_val + 5. / 6. * in[0];
  out[1] = 5. / 6. * in[1] + 1. / 6. * in[2];
  out[2] = 0.5 * in[2] + 0.5 * in[3];
}

void GetSubframesPitchParameters(int sampling_rate_hz,
                                 double* gains,
                                 double* lags,
                                 int num_in_frames,
                                 int num_out_frames,
                                 double* log_old_gain,
                                 double* old_lag,
                                 double* log_pitch_gain,
                                 double* pitch_lag_hz) {
  // Gain interpolation is in log-domain, also returned in log-domain.
  for (int n = 0; n < num_in_frames; n++)
    gains[n] = log(gains[n] + 1e-12);

  // Interpolate lags and gains.
  PitchInterpolation(*log_old_gain, gains, log_pitch_gain);
  *log_old_gain = gains[num_in_frames - 1];
  PitchInterpolation(*old_lag, lags, pitch_lag_hz);
  *old_lag = lags[num_in_frames - 1];

  // Convert pitch-lags to Hertz.
  for (int n = 0; n < num_out_frames; n++) {
    pitch_lag_hz[n] = (sampling_rate_hz) / (pitch_lag_hz[n]);
  }
}

}  // namespace webrtc
