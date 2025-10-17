/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 15, 2023.
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
#ifndef MODULES_AUDIO_PROCESSING_AEC3_MOCK_MOCK_RENDER_DELAY_BUFFER_H_
#define MODULES_AUDIO_PROCESSING_AEC3_MOCK_MOCK_RENDER_DELAY_BUFFER_H_

#include <vector>

#include "modules/audio_processing/aec3/aec3_common.h"
#include "modules/audio_processing/aec3/downsampled_render_buffer.h"
#include "modules/audio_processing/aec3/render_buffer.h"
#include "modules/audio_processing/aec3/render_delay_buffer.h"
#include "test/gmock.h"

namespace webrtc {
namespace test {

class MockRenderDelayBuffer : public RenderDelayBuffer {
 public:
  MockRenderDelayBuffer(int sample_rate_hz, size_t num_channels);
  virtual ~MockRenderDelayBuffer();

  MOCK_METHOD(void, Reset, (), (override));
  MOCK_METHOD(RenderDelayBuffer::BufferingEvent,
              Insert,
              (const Block& block),
              (override));
  MOCK_METHOD(void, HandleSkippedCaptureProcessing, (), (override));
  MOCK_METHOD(RenderDelayBuffer::BufferingEvent,
              PrepareCaptureProcessing,
              (),
              (override));
  MOCK_METHOD(bool, AlignFromDelay, (size_t delay), (override));
  MOCK_METHOD(void, AlignFromExternalDelay, (), (override));
  MOCK_METHOD(size_t, Delay, (), (const, override));
  MOCK_METHOD(size_t, MaxDelay, (), (const, override));
  MOCK_METHOD(RenderBuffer*, GetRenderBuffer, (), (override));
  MOCK_METHOD(const DownsampledRenderBuffer&,
              GetDownsampledRenderBuffer,
              (),
              (const, override));
  MOCK_METHOD(void, SetAudioBufferDelay, (int delay_ms), (override));
  MOCK_METHOD(bool, HasReceivedBufferDelay, (), (override));

 private:
  RenderBuffer* FakeGetRenderBuffer() { return &render_buffer_; }
  const DownsampledRenderBuffer& FakeGetDownsampledRenderBuffer() const {
    return downsampled_render_buffer_;
  }
  BlockBuffer block_buffer_;
  SpectrumBuffer spectrum_buffer_;
  FftBuffer fft_buffer_;
  RenderBuffer render_buffer_;
  DownsampledRenderBuffer downsampled_render_buffer_;
};

}  // namespace test
}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AEC3_MOCK_MOCK_RENDER_DELAY_BUFFER_H_
