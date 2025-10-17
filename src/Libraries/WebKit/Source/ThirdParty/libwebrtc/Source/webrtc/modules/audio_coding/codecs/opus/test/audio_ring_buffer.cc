/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 29, 2024.
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
#include "modules/audio_coding/codecs/opus/test/audio_ring_buffer.h"

#include "common_audio/ring_buffer.h"
#include "rtc_base/checks.h"

// This is a simple multi-channel wrapper over the ring_buffer.h C interface.

namespace webrtc {

AudioRingBuffer::AudioRingBuffer(size_t channels, size_t max_frames) {
  buffers_.reserve(channels);
  for (size_t i = 0; i < channels; ++i)
    buffers_.push_back(WebRtc_CreateBuffer(max_frames, sizeof(float)));
}

AudioRingBuffer::~AudioRingBuffer() {
  for (auto* buf : buffers_)
    WebRtc_FreeBuffer(buf);
}

void AudioRingBuffer::Write(const float* const* data,
                            size_t channels,
                            size_t frames) {
  RTC_DCHECK_EQ(buffers_.size(), channels);
  for (size_t i = 0; i < channels; ++i) {
    const size_t written = WebRtc_WriteBuffer(buffers_[i], data[i], frames);
    RTC_CHECK_EQ(written, frames);
  }
}

void AudioRingBuffer::Read(float* const* data, size_t channels, size_t frames) {
  RTC_DCHECK_EQ(buffers_.size(), channels);
  for (size_t i = 0; i < channels; ++i) {
    const size_t read =
        WebRtc_ReadBuffer(buffers_[i], nullptr, data[i], frames);
    RTC_CHECK_EQ(read, frames);
  }
}

size_t AudioRingBuffer::ReadFramesAvailable() const {
  // All buffers have the same amount available.
  return WebRtc_available_read(buffers_[0]);
}

size_t AudioRingBuffer::WriteFramesAvailable() const {
  // All buffers have the same amount available.
  return WebRtc_available_write(buffers_[0]);
}

void AudioRingBuffer::MoveReadPositionForward(size_t frames) {
  for (auto* buf : buffers_) {
    const size_t moved =
        static_cast<size_t>(WebRtc_MoveReadPtr(buf, static_cast<int>(frames)));
    RTC_CHECK_EQ(moved, frames);
  }
}

void AudioRingBuffer::MoveReadPositionBackward(size_t frames) {
  for (auto* buf : buffers_) {
    const size_t moved = static_cast<size_t>(
        -WebRtc_MoveReadPtr(buf, -static_cast<int>(frames)));
    RTC_CHECK_EQ(moved, frames);
  }
}

}  // namespace webrtc
