/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 5, 2022.
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
#include "modules/audio_processing/aec3/mock/mock_render_delay_buffer.h"

namespace webrtc {
namespace test {

MockRenderDelayBuffer::MockRenderDelayBuffer(int sample_rate_hz,
                                             size_t num_channels)
    : block_buffer_(GetRenderDelayBufferSize(4, 4, 12),
                    NumBandsForRate(sample_rate_hz),
                    num_channels),
      spectrum_buffer_(block_buffer_.buffer.size(), num_channels),
      fft_buffer_(block_buffer_.buffer.size(), num_channels),
      render_buffer_(&block_buffer_, &spectrum_buffer_, &fft_buffer_),
      downsampled_render_buffer_(GetDownSampledBufferSize(4, 4)) {
  ON_CALL(*this, GetRenderBuffer())
      .WillByDefault(
          ::testing::Invoke(this, &MockRenderDelayBuffer::FakeGetRenderBuffer));
  ON_CALL(*this, GetDownsampledRenderBuffer())
      .WillByDefault(::testing::Invoke(
          this, &MockRenderDelayBuffer::FakeGetDownsampledRenderBuffer));
}

MockRenderDelayBuffer::~MockRenderDelayBuffer() = default;

}  // namespace test
}  // namespace webrtc
