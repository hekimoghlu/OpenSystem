/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 21, 2022.
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
#ifndef MODULES_AUDIO_CODING_CODECS_OPUS_TEST_AUDIO_RING_BUFFER_H_
#define MODULES_AUDIO_CODING_CODECS_OPUS_TEST_AUDIO_RING_BUFFER_H_

#include <stddef.h>

#include <memory>
#include <vector>

struct RingBuffer;

namespace webrtc {

// A ring buffer tailored for float deinterleaved audio. Any operation that
// cannot be performed as requested will cause a crash (e.g. insufficient data
// in the buffer to fulfill a read request.)
class AudioRingBuffer final {
 public:
  // Specify the number of channels and maximum number of frames the buffer will
  // contain.
  AudioRingBuffer(size_t channels, size_t max_frames);
  ~AudioRingBuffer();

  // Copies `data` to the buffer and advances the write pointer. `channels` must
  // be the same as at creation time.
  void Write(const float* const* data, size_t channels, size_t frames);

  // Copies from the buffer to `data` and advances the read pointer. `channels`
  // must be the same as at creation time.
  void Read(float* const* data, size_t channels, size_t frames);

  size_t ReadFramesAvailable() const;
  size_t WriteFramesAvailable() const;

  // Moves the read position. The forward version advances the read pointer
  // towards the write pointer and the backward verison withdraws the read
  // pointer away from the write pointer (i.e. flushing and stuffing the buffer
  // respectively.)
  void MoveReadPositionForward(size_t frames);
  void MoveReadPositionBackward(size_t frames);

 private:
  // TODO(kwiberg): Use std::vector<std::unique_ptr<RingBuffer>> instead.
  std::vector<RingBuffer*> buffers_;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_CODING_CODECS_OPUS_TEST_AUDIO_RING_BUFFER_H_
