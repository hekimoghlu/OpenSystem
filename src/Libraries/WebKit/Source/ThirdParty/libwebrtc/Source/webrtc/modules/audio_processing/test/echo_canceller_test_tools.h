/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 8, 2021.
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
#ifndef MODULES_AUDIO_PROCESSING_TEST_ECHO_CANCELLER_TEST_TOOLS_H_
#define MODULES_AUDIO_PROCESSING_TEST_ECHO_CANCELLER_TEST_TOOLS_H_

#include <algorithm>
#include <vector>

#include "api/array_view.h"
#include "rtc_base/random.h"

namespace webrtc {

// Randomizes the elements in a vector with values -32767.f:32767.f.
void RandomizeSampleVector(Random* random_generator, rtc::ArrayView<float> v);

// Randomizes the elements in a vector with values -amplitude:amplitude.
void RandomizeSampleVector(Random* random_generator,
                           rtc::ArrayView<float> v,
                           float amplitude);

// Class for delaying a signal a fixed number of samples.
template <typename T>
class DelayBuffer {
 public:
  explicit DelayBuffer(size_t delay) : buffer_(delay) {}
  ~DelayBuffer() = default;

  // Produces a delayed signal copy of x.
  void Delay(rtc::ArrayView<const T> x, rtc::ArrayView<T> x_delayed);

 private:
  std::vector<T> buffer_;
  size_t next_insert_index_ = 0;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_TEST_ECHO_CANCELLER_TEST_TOOLS_H_
