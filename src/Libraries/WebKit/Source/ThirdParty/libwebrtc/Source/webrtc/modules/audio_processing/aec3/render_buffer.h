/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 24, 2023.
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
#ifndef MODULES_AUDIO_PROCESSING_AEC3_RENDER_BUFFER_H_
#define MODULES_AUDIO_PROCESSING_AEC3_RENDER_BUFFER_H_

#include <stddef.h>

#include <array>
#include <vector>

#include "api/array_view.h"
#include "modules/audio_processing/aec3/aec3_common.h"
#include "modules/audio_processing/aec3/block_buffer.h"
#include "modules/audio_processing/aec3/fft_buffer.h"
#include "modules/audio_processing/aec3/fft_data.h"
#include "modules/audio_processing/aec3/spectrum_buffer.h"
#include "rtc_base/checks.h"

namespace webrtc {

// Provides a buffer of the render data for the echo remover.
class RenderBuffer {
 public:
  RenderBuffer(BlockBuffer* block_buffer,
               SpectrumBuffer* spectrum_buffer,
               FftBuffer* fft_buffer);

  RenderBuffer() = delete;
  RenderBuffer(const RenderBuffer&) = delete;
  RenderBuffer& operator=(const RenderBuffer&) = delete;

  ~RenderBuffer();

  // Get a block.
  const Block& GetBlock(int buffer_offset_blocks) const {
    int position =
        block_buffer_->OffsetIndex(block_buffer_->read, buffer_offset_blocks);
    return block_buffer_->buffer[position];
  }

  // Get the spectrum from one of the FFTs in the buffer.
  rtc::ArrayView<const std::array<float, kFftLengthBy2Plus1>> Spectrum(
      int buffer_offset_ffts) const {
    int position = spectrum_buffer_->OffsetIndex(spectrum_buffer_->read,
                                                 buffer_offset_ffts);
    return spectrum_buffer_->buffer[position];
  }

  // Returns the circular fft buffer.
  rtc::ArrayView<const std::vector<FftData>> GetFftBuffer() const {
    return fft_buffer_->buffer;
  }

  // Returns the current position in the circular buffer.
  size_t Position() const {
    RTC_DCHECK_EQ(spectrum_buffer_->read, fft_buffer_->read);
    RTC_DCHECK_EQ(spectrum_buffer_->write, fft_buffer_->write);
    return fft_buffer_->read;
  }

  // Returns the sum of the spectrums for a certain number of FFTs.
  void SpectralSum(size_t num_spectra,
                   std::array<float, kFftLengthBy2Plus1>* X2) const;

  // Returns the sums of the spectrums for two numbers of FFTs.
  void SpectralSums(size_t num_spectra_shorter,
                    size_t num_spectra_longer,
                    std::array<float, kFftLengthBy2Plus1>* X2_shorter,
                    std::array<float, kFftLengthBy2Plus1>* X2_longer) const;

  // Gets the recent activity seen in the render signal.
  bool GetRenderActivity() const { return render_activity_; }

  // Specifies the recent activity seen in the render signal.
  void SetRenderActivity(bool activity) { render_activity_ = activity; }

  // Returns the headroom between the write and the read positions in the
  // buffer.
  int Headroom() const {
    // The write and read indices are decreased over time.
    int headroom =
        fft_buffer_->write < fft_buffer_->read
            ? fft_buffer_->read - fft_buffer_->write
            : fft_buffer_->size - fft_buffer_->write + fft_buffer_->read;

    RTC_DCHECK_LE(0, headroom);
    RTC_DCHECK_GE(fft_buffer_->size, headroom);

    return headroom;
  }

  // Returns a reference to the spectrum buffer.
  const SpectrumBuffer& GetSpectrumBuffer() const { return *spectrum_buffer_; }

  // Returns a reference to the block buffer.
  const BlockBuffer& GetBlockBuffer() const { return *block_buffer_; }

 private:
  const BlockBuffer* const block_buffer_;
  const SpectrumBuffer* const spectrum_buffer_;
  const FftBuffer* const fft_buffer_;
  bool render_activity_ = false;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AEC3_RENDER_BUFFER_H_
