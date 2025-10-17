/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 1, 2022.
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
#ifndef MODULES_AUDIO_PROCESSING_AGC2_SATURATION_PROTECTOR_BUFFER_H_
#define MODULES_AUDIO_PROCESSING_AGC2_SATURATION_PROTECTOR_BUFFER_H_

#include <array>
#include <optional>

#include "modules/audio_processing/agc2/agc2_common.h"

namespace webrtc {

// Ring buffer for the saturation protector which only supports (i) push back
// and (ii) read oldest item.
class SaturationProtectorBuffer {
 public:
  SaturationProtectorBuffer();
  ~SaturationProtectorBuffer();

  bool operator==(const SaturationProtectorBuffer& b) const;
  inline bool operator!=(const SaturationProtectorBuffer& b) const {
    return !(*this == b);
  }

  // Maximum number of values that the buffer can contain.
  int Capacity() const;

  // Number of values in the buffer.
  int Size() const;

  void Reset();

  // Pushes back `v`. If the buffer is full, the oldest value is replaced.
  void PushBack(float v);

  // Returns the oldest item in the buffer. Returns an empty value if the
  // buffer is empty.
  std::optional<float> Front() const;

 private:
  int FrontIndex() const;
  // `buffer_` has `size_` elements (up to the size of `buffer_`) and `next_` is
  // the position where the next new value is written in `buffer_`.
  std::array<float, kSaturationProtectorBufferSize> buffer_;
  int next_ = 0;
  int size_ = 0;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AGC2_SATURATION_PROTECTOR_BUFFER_H_
