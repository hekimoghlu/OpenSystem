/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 3, 2022.
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
#include "common_audio/signal_processing/include/real_fft.h"

#include <stdlib.h>

#include "common_audio/signal_processing/include/signal_processing_library.h"

struct RealFFT {
  int order;
};

struct RealFFT* WebRtcSpl_CreateRealFFT(int order) {
  struct RealFFT* self = NULL;

  if (order > kMaxFFTOrder || order < 0) {
    return NULL;
  }

  self = malloc(sizeof(struct RealFFT));
  if (self == NULL) {
    return NULL;
  }
  self->order = order;

  return self;
}

void WebRtcSpl_FreeRealFFT(struct RealFFT* self) {
  if (self != NULL) {
    free(self);
  }
}

// The C version FFT functions (i.e. WebRtcSpl_RealForwardFFT and
// WebRtcSpl_RealInverseFFT) are real-valued FFT wrappers for complex-valued
// FFT implementation in SPL.

int WebRtcSpl_RealForwardFFT(struct RealFFT* self,
                             const int16_t* real_data_in,
                             int16_t* complex_data_out) {
  int i = 0;
  int j = 0;
  int result = 0;
  int n = 1 << self->order;
  // The complex-value FFT implementation needs a buffer to hold 2^order
  // 16-bit COMPLEX numbers, for both time and frequency data.
  int16_t complex_buffer[2 << kMaxFFTOrder];

  // Insert zeros to the imaginary parts for complex forward FFT input.
  for (i = 0, j = 0; i < n; i += 1, j += 2) {
    complex_buffer[j] = real_data_in[i];
    complex_buffer[j + 1] = 0;
  };

  WebRtcSpl_ComplexBitReverse(complex_buffer, self->order);
  result = WebRtcSpl_ComplexFFT(complex_buffer, self->order, 1);

  // For real FFT output, use only the first N + 2 elements from
  // complex forward FFT.
  memcpy(complex_data_out, complex_buffer, sizeof(int16_t) * (n + 2));

  return result;
}

int WebRtcSpl_RealInverseFFT(struct RealFFT* self,
                             const int16_t* complex_data_in,
                             int16_t* real_data_out) {
  int i = 0;
  int j = 0;
  int result = 0;
  int n = 1 << self->order;
  // Create the buffer specific to complex-valued FFT implementation.
  int16_t complex_buffer[2 << kMaxFFTOrder];

  // For n-point FFT, first copy the first n + 2 elements into complex
  // FFT, then construct the remaining n - 2 elements by real FFT's
  // conjugate-symmetric properties.
  memcpy(complex_buffer, complex_data_in, sizeof(int16_t) * (n + 2));
  for (i = n + 2; i < 2 * n; i += 2) {
    complex_buffer[i] = complex_data_in[2 * n - i];
    complex_buffer[i + 1] = -complex_data_in[2 * n - i + 1];
  }

  WebRtcSpl_ComplexBitReverse(complex_buffer, self->order);
  result = WebRtcSpl_ComplexIFFT(complex_buffer, self->order, 1);

  // Strip out the imaginary parts of the complex inverse FFT output.
  for (i = 0, j = 0; i < n; i += 1, j += 2) {
    real_data_out[i] = complex_buffer[j];
  }

  return result;
}
