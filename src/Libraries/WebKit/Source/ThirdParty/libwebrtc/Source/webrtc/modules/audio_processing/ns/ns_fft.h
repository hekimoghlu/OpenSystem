/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 8, 2024.
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
#ifndef MODULES_AUDIO_PROCESSING_NS_NS_FFT_H_
#define MODULES_AUDIO_PROCESSING_NS_NS_FFT_H_

#include <vector>

#include "api/array_view.h"
#include "modules/audio_processing/ns/ns_common.h"

namespace webrtc {

// Wrapper class providing 256 point FFT functionality.
class NrFft {
 public:
  NrFft();
  NrFft(const NrFft&) = delete;
  NrFft& operator=(const NrFft&) = delete;

  // Transforms the signal from time to frequency domain.
  void Fft(rtc::ArrayView<float, kFftSize> time_data,
           rtc::ArrayView<float, kFftSize> real,
           rtc::ArrayView<float, kFftSize> imag);

  // Transforms the signal from frequency to time domain.
  void Ifft(rtc::ArrayView<const float> real,
            rtc::ArrayView<const float> imag,
            rtc::ArrayView<float> time_data);

 private:
  std::vector<size_t> bit_reversal_state_;
  std::vector<float> tables_;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_NS_NS_FFT_H_
