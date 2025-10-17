/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 3, 2022.
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
#ifndef COMMON_AUDIO_REAL_FOURIER_OOURA_H_
#define COMMON_AUDIO_REAL_FOURIER_OOURA_H_

#include <stddef.h>

#include <complex>
#include <memory>

#include "common_audio/real_fourier.h"

namespace webrtc {

class RealFourierOoura : public RealFourier {
 public:
  explicit RealFourierOoura(int fft_order);
  ~RealFourierOoura() override;

  void Forward(const float* src, std::complex<float>* dest) const override;
  void Inverse(const std::complex<float>* src, float* dest) const override;

  int order() const override;

 private:
  const int order_;
  const size_t length_;
  const size_t complex_length_;
  // These are work arrays for Ooura. The names are based on the comments in
  // common_audio/third_party/ooura/fft_size_256/fft4g.cc.
  const std::unique_ptr<size_t[]> work_ip_;
  const std::unique_ptr<float[]> work_w_;
};

}  // namespace webrtc

#endif  // COMMON_AUDIO_REAL_FOURIER_OOURA_H_
