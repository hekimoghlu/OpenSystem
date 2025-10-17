/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 27, 2022.
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
#ifndef MODULES_AUDIO_PROCESSING_AEC3_AEC3_FFT_H_
#define MODULES_AUDIO_PROCESSING_AEC3_AEC3_FFT_H_

#include <array>

#include "api/array_view.h"
#include "common_audio/third_party/ooura/fft_size_128/ooura_fft.h"
#include "modules/audio_processing/aec3/aec3_common.h"
#include "modules/audio_processing/aec3/fft_data.h"
#include "rtc_base/checks.h"

namespace webrtc {

// Wrapper class that provides 128 point real valued FFT functionality with the
// FftData type.
class Aec3Fft {
 public:
  enum class Window { kRectangular, kHanning, kSqrtHanning };

  Aec3Fft();

  Aec3Fft(const Aec3Fft&) = delete;
  Aec3Fft& operator=(const Aec3Fft&) = delete;

  // Computes the FFT. Note that both the input and output are modified.
  void Fft(std::array<float, kFftLength>* x, FftData* X) const {
    RTC_DCHECK(x);
    RTC_DCHECK(X);
    ooura_fft_.Fft(x->data());
    X->CopyFromPackedArray(*x);
  }
  // Computes the inverse Fft.
  void Ifft(const FftData& X, std::array<float, kFftLength>* x) const {
    RTC_DCHECK(x);
    X.CopyToPackedArray(x);
    ooura_fft_.InverseFft(x->data());
  }

  // Windows the input using a Hanning window, and then adds padding of
  // kFftLengthBy2 initial zeros before computing the Fft.
  void ZeroPaddedFft(rtc::ArrayView<const float> x,
                     Window window,
                     FftData* X) const;

  // Concatenates the kFftLengthBy2 values long x and x_old before computing the
  // Fft. After that, x is copied to x_old.
  void PaddedFft(rtc::ArrayView<const float> x,
                 rtc::ArrayView<const float> x_old,
                 FftData* X) const {
    PaddedFft(x, x_old, Window::kRectangular, X);
  }

  // Padded Fft using a time-domain window.
  void PaddedFft(rtc::ArrayView<const float> x,
                 rtc::ArrayView<const float> x_old,
                 Window window,
                 FftData* X) const;

 private:
  const OouraFft ooura_fft_;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AEC3_AEC3_FFT_H_
