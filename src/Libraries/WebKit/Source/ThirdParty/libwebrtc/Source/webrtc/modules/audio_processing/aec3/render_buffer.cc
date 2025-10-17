/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 16, 2025.
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
#include "modules/audio_processing/aec3/render_buffer.h"

#include <algorithm>
#include <functional>

#include "modules/audio_processing/aec3/aec3_common.h"
#include "rtc_base/checks.h"

namespace webrtc {

RenderBuffer::RenderBuffer(BlockBuffer* block_buffer,
                           SpectrumBuffer* spectrum_buffer,
                           FftBuffer* fft_buffer)
    : block_buffer_(block_buffer),
      spectrum_buffer_(spectrum_buffer),
      fft_buffer_(fft_buffer) {
  RTC_DCHECK(block_buffer_);
  RTC_DCHECK(spectrum_buffer_);
  RTC_DCHECK(fft_buffer_);
  RTC_DCHECK_EQ(block_buffer_->buffer.size(), fft_buffer_->buffer.size());
  RTC_DCHECK_EQ(spectrum_buffer_->buffer.size(), fft_buffer_->buffer.size());
  RTC_DCHECK_EQ(spectrum_buffer_->read, fft_buffer_->read);
  RTC_DCHECK_EQ(spectrum_buffer_->write, fft_buffer_->write);
}

RenderBuffer::~RenderBuffer() = default;

void RenderBuffer::SpectralSum(
    size_t num_spectra,
    std::array<float, kFftLengthBy2Plus1>* X2) const {
  X2->fill(0.f);
  int position = spectrum_buffer_->read;
  for (size_t j = 0; j < num_spectra; ++j) {
    for (const auto& channel_spectrum : spectrum_buffer_->buffer[position]) {
      for (size_t k = 0; k < X2->size(); ++k) {
        (*X2)[k] += channel_spectrum[k];
      }
    }
    position = spectrum_buffer_->IncIndex(position);
  }
}

void RenderBuffer::SpectralSums(
    size_t num_spectra_shorter,
    size_t num_spectra_longer,
    std::array<float, kFftLengthBy2Plus1>* X2_shorter,
    std::array<float, kFftLengthBy2Plus1>* X2_longer) const {
  RTC_DCHECK_LE(num_spectra_shorter, num_spectra_longer);
  X2_shorter->fill(0.f);
  int position = spectrum_buffer_->read;
  size_t j = 0;
  for (; j < num_spectra_shorter; ++j) {
    for (const auto& channel_spectrum : spectrum_buffer_->buffer[position]) {
      for (size_t k = 0; k < X2_shorter->size(); ++k) {
        (*X2_shorter)[k] += channel_spectrum[k];
      }
    }
    position = spectrum_buffer_->IncIndex(position);
  }
  std::copy(X2_shorter->begin(), X2_shorter->end(), X2_longer->begin());
  for (; j < num_spectra_longer; ++j) {
    for (const auto& channel_spectrum : spectrum_buffer_->buffer[position]) {
      for (size_t k = 0; k < X2_longer->size(); ++k) {
        (*X2_longer)[k] += channel_spectrum[k];
      }
    }
    position = spectrum_buffer_->IncIndex(position);
  }
}

}  // namespace webrtc
