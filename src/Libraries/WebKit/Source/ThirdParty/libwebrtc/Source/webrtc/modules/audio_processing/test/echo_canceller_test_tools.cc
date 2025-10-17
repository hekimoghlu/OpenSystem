/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 7, 2024.
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
#include "modules/audio_processing/test/echo_canceller_test_tools.h"

#include "rtc_base/checks.h"

namespace webrtc {

void RandomizeSampleVector(Random* random_generator, rtc::ArrayView<float> v) {
  RandomizeSampleVector(random_generator, v,
                        /*amplitude=*/32767.f);
}

void RandomizeSampleVector(Random* random_generator,
                           rtc::ArrayView<float> v,
                           float amplitude) {
  for (auto& v_k : v) {
    v_k = 2 * amplitude * random_generator->Rand<float>() - amplitude;
  }
}

template <typename T>
void DelayBuffer<T>::Delay(rtc::ArrayView<const T> x,
                           rtc::ArrayView<T> x_delayed) {
  RTC_DCHECK_EQ(x.size(), x_delayed.size());
  if (buffer_.empty()) {
    std::copy(x.begin(), x.end(), x_delayed.begin());
  } else {
    for (size_t k = 0; k < x.size(); ++k) {
      x_delayed[k] = buffer_[next_insert_index_];
      buffer_[next_insert_index_] = x[k];
      next_insert_index_ = (next_insert_index_ + 1) % buffer_.size();
    }
  }
}

template class DelayBuffer<float>;
template class DelayBuffer<int>;
}  // namespace webrtc
