/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 24, 2024.
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
#ifndef MODULES_AUDIO_PROCESSING_ECHO_DETECTOR_CIRCULAR_BUFFER_H_
#define MODULES_AUDIO_PROCESSING_ECHO_DETECTOR_CIRCULAR_BUFFER_H_

#include <stddef.h>

#include <optional>
#include <vector>

namespace webrtc {

// Ring buffer containing floating point values.
struct CircularBuffer {
 public:
  explicit CircularBuffer(size_t size);
  ~CircularBuffer();

  void Push(float value);
  std::optional<float> Pop();
  size_t Size() const { return nr_elements_in_buffer_; }
  // This function fills the buffer with zeros, but does not change its size.
  void Clear();

 private:
  std::vector<float> buffer_;
  size_t next_insertion_index_ = 0;
  // This is the number of elements that have been pushed into the circular
  // buffer, not the allocated buffer size.
  size_t nr_elements_in_buffer_ = 0;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_ECHO_DETECTOR_CIRCULAR_BUFFER_H_
