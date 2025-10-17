/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 5, 2022.
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
#ifndef MODULES_AUDIO_PROCESSING_AGC2_RNN_VAD_AUTO_CORRELATION_H_
#define MODULES_AUDIO_PROCESSING_AGC2_RNN_VAD_AUTO_CORRELATION_H_

#include <memory>

#include "api/array_view.h"
#include "modules/audio_processing/agc2/rnn_vad/common.h"
#include "modules/audio_processing/utility/pffft_wrapper.h"

namespace webrtc {
namespace rnn_vad {

// Class to compute the auto correlation on the pitch buffer for a target pitch
// interval.
class AutoCorrelationCalculator {
 public:
  AutoCorrelationCalculator();
  AutoCorrelationCalculator(const AutoCorrelationCalculator&) = delete;
  AutoCorrelationCalculator& operator=(const AutoCorrelationCalculator&) =
      delete;
  ~AutoCorrelationCalculator();

  // Computes the auto-correlation coefficients for a target pitch interval.
  // `auto_corr` indexes are inverted lags.
  void ComputeOnPitchBuffer(
      rtc::ArrayView<const float, kBufSize12kHz> pitch_buf,
      rtc::ArrayView<float, kNumLags12kHz> auto_corr);

 private:
  Pffft fft_;
  std::unique_ptr<Pffft::FloatBuffer> tmp_;
  std::unique_ptr<Pffft::FloatBuffer> X_;
  std::unique_ptr<Pffft::FloatBuffer> H_;
};

}  // namespace rnn_vad
}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AGC2_RNN_VAD_AUTO_CORRELATION_H_
