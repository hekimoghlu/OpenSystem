/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 25, 2023.
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
#ifndef MODULES_AUDIO_PROCESSING_UTILITY_PFFFT_WRAPPER_H_
#define MODULES_AUDIO_PROCESSING_UTILITY_PFFFT_WRAPPER_H_

#include <memory>

#include "api/array_view.h"

// Forward declaration.
struct PFFFT_Setup;

namespace webrtc {

// Pretty-Fast Fast Fourier Transform (PFFFT) wrapper class.
// Not thread safe.
class Pffft {
 public:
  enum class FftType { kReal, kComplex };

  // 1D floating point buffer used as input/output data type for the FFT ops.
  // It must be constructed using Pffft::CreateBuffer().
  class FloatBuffer {
   public:
    FloatBuffer(const FloatBuffer&) = delete;
    FloatBuffer& operator=(const FloatBuffer&) = delete;
    ~FloatBuffer();

    rtc::ArrayView<const float> GetConstView() const;
    rtc::ArrayView<float> GetView();

   private:
    friend class Pffft;
    FloatBuffer(size_t fft_size, FftType fft_type);
    const float* const_data() const { return data_; }
    float* data() { return data_; }
    size_t size() const { return size_; }

    const size_t size_;
    float* const data_;
  };

  // TODO(https://crbug.com/webrtc/9577): Consider adding a factory and making
  // the ctor private.
  // static std::unique_ptr<Pffft> Create(size_t fft_size,
  // FftType fft_type); Ctor. `fft_size` must be a supported size (see
  // Pffft::IsValidFftSize()). If not supported, the code will crash.
  Pffft(size_t fft_size, FftType fft_type);
  Pffft(const Pffft&) = delete;
  Pffft& operator=(const Pffft&) = delete;
  ~Pffft();

  // Returns true if the FFT size is supported.
  static bool IsValidFftSize(size_t fft_size, FftType fft_type);

  // Returns true if SIMD code optimizations are being used.
  static bool IsSimdEnabled();

  // Creates a buffer of the right size.
  std::unique_ptr<FloatBuffer> CreateBuffer() const;

  // TODO(https://crbug.com/webrtc/9577): Overload with rtc::ArrayView args.
  // Computes the forward fast Fourier transform.
  void ForwardTransform(const FloatBuffer& in, FloatBuffer* out, bool ordered);
  // Computes the backward fast Fourier transform.
  void BackwardTransform(const FloatBuffer& in, FloatBuffer* out, bool ordered);

  // Multiplies the frequency components of `fft_x` and `fft_y` and accumulates
  // them into `out`. The arrays must have been obtained with
  // ForwardTransform(..., /*ordered=*/false) - i.e., `fft_x` and `fft_y` must
  // not be ordered.
  void FrequencyDomainConvolve(const FloatBuffer& fft_x,
                               const FloatBuffer& fft_y,
                               FloatBuffer* out,
                               float scaling = 1.f);

 private:
  const size_t fft_size_;
  const FftType fft_type_;
  PFFFT_Setup* pffft_status_;
  float* const scratch_buffer_;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_UTILITY_PFFFT_WRAPPER_H_
