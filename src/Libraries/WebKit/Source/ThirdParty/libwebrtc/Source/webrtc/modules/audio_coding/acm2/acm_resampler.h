/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 1, 2024.
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
#ifndef MODULES_AUDIO_CODING_ACM2_ACM_RESAMPLER_H_
#define MODULES_AUDIO_CODING_ACM2_ACM_RESAMPLER_H_

#include <stddef.h>
#include <stdint.h>

#include "api/audio/audio_frame.h"
#include "common_audio/resampler/include/push_resampler.h"

namespace webrtc {
namespace acm2 {

class ACMResampler {
 public:
  ACMResampler();
  ~ACMResampler();

  // TODO: b/335805780 - Change to accept InterleavedView<>.
  int Resample10Msec(const int16_t* in_audio,
                     int in_freq_hz,
                     int out_freq_hz,
                     size_t num_audio_channels,
                     size_t out_capacity_samples,
                     int16_t* out_audio);

 private:
  PushResampler<int16_t> resampler_;
};

// Helper class to perform resampling if needed, meant to be used after
// receiving the audio_frame from NetEq. Provides reasonably glitch free
// transitions between different output sample rates from NetEq.
class ResamplerHelper {
 public:
  ResamplerHelper();

  // Resamples audio_frame if it is not already in desired_sample_rate_hz.
  bool MaybeResample(int desired_sample_rate_hz, AudioFrame* audio_frame);

 private:
  ACMResampler resampler_;
  bool resampled_last_output_frame_ = true;
  std::array<int16_t, AudioFrame::kMaxDataSizeSamples> last_audio_buffer_;
};

}  // namespace acm2
}  // namespace webrtc

#endif  // MODULES_AUDIO_CODING_ACM2_ACM_RESAMPLER_H_
