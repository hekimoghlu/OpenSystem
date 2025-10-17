/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 15, 2022.
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
#include "modules/audio_processing/ns/ns_fft.h"

#include "common_audio/third_party/ooura/fft_size_256/fft4g.h"

namespace webrtc {

NrFft::NrFft() : bit_reversal_state_(kFftSize / 2), tables_(kFftSize / 2) {
  // Initialize WebRtc_rdt (setting (bit_reversal_state_[0] to 0 triggers
  // initialization)
  bit_reversal_state_[0] = 0.f;
  std::array<float, kFftSize> tmp_buffer;
  tmp_buffer.fill(0.f);
  WebRtc_rdft(kFftSize, 1, tmp_buffer.data(), bit_reversal_state_.data(),
              tables_.data());
}

void NrFft::Fft(rtc::ArrayView<float, kFftSize> time_data,
                rtc::ArrayView<float, kFftSize> real,
                rtc::ArrayView<float, kFftSize> imag) {
  WebRtc_rdft(kFftSize, 1, time_data.data(), bit_reversal_state_.data(),
              tables_.data());

  imag[0] = 0;
  real[0] = time_data[0];

  imag[kFftSizeBy2Plus1 - 1] = 0;
  real[kFftSizeBy2Plus1 - 1] = time_data[1];

  for (size_t i = 1; i < kFftSizeBy2Plus1 - 1; ++i) {
    real[i] = time_data[2 * i];
    imag[i] = time_data[2 * i + 1];
  }
}

void NrFft::Ifft(rtc::ArrayView<const float> real,
                 rtc::ArrayView<const float> imag,
                 rtc::ArrayView<float> time_data) {
  time_data[0] = real[0];
  time_data[1] = real[kFftSizeBy2Plus1 - 1];
  for (size_t i = 1; i < kFftSizeBy2Plus1 - 1; ++i) {
    time_data[2 * i] = real[i];
    time_data[2 * i + 1] = imag[i];
  }
  WebRtc_rdft(kFftSize, -1, time_data.data(), bit_reversal_state_.data(),
              tables_.data());

  // Scale the output
  constexpr float kScaling = 2.f / kFftSize;
  for (float& d : time_data) {
    d *= kScaling;
  }
}

}  // namespace webrtc
