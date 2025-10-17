/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 16, 2025.
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
#ifndef MODULES_AUDIO_PROCESSING_AGC2_RNN_VAD_SEQUENCE_BUFFER_H_
#define MODULES_AUDIO_PROCESSING_AGC2_RNN_VAD_SEQUENCE_BUFFER_H_

#include <algorithm>
#include <cstring>
#include <type_traits>
#include <vector>

#include "api/array_view.h"
#include "rtc_base/checks.h"

namespace webrtc {
namespace rnn_vad {

// Linear buffer implementation to (i) push fixed size chunks of sequential data
// and (ii) view contiguous parts of the buffer. The buffer and the pushed
// chunks have size S and N respectively. For instance, when S = 2N the first
// half of the sequence buffer is replaced with its second half, and the new N
// values are written at the end of the buffer.
// The class also provides a view on the most recent M values, where 0 < M <= S
// and by default M = N.
template <typename T, int S, int N, int M = N>
class SequenceBuffer {
  static_assert(N <= S,
                "The new chunk size cannot be larger than the sequence buffer "
                "size.");
  static_assert(std::is_arithmetic<T>::value,
                "Integral or floating point required.");

 public:
  SequenceBuffer() : buffer_(S) {
    RTC_DCHECK_EQ(S, buffer_.size());
    Reset();
  }
  SequenceBuffer(const SequenceBuffer&) = delete;
  SequenceBuffer& operator=(const SequenceBuffer&) = delete;
  ~SequenceBuffer() = default;
  int size() const { return S; }
  int chunks_size() const { return N; }
  // Sets the sequence buffer values to zero.
  void Reset() { std::fill(buffer_.begin(), buffer_.end(), 0); }
  // Returns a view on the whole buffer.
  rtc::ArrayView<const T, S> GetBufferView() const {
    return {buffer_.data(), S};
  }
  // Returns a view on the M most recent values of the buffer.
  rtc::ArrayView<const T, M> GetMostRecentValuesView() const {
    static_assert(M <= S,
                  "The number of most recent values cannot be larger than the "
                  "sequence buffer size.");
    return {buffer_.data() + S - M, M};
  }
  // Shifts left the buffer by N items and add new N items at the end.
  void Push(rtc::ArrayView<const T, N> new_values) {
    // Make space for the new values.
    if (S > N)
      std::memmove(buffer_.data(), buffer_.data() + N, (S - N) * sizeof(T));
    // Copy the new values at the end of the buffer.
    std::memcpy(buffer_.data() + S - N, new_values.data(), N * sizeof(T));
  }

 private:
  std::vector<T> buffer_;
};

}  // namespace rnn_vad
}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AGC2_RNN_VAD_SEQUENCE_BUFFER_H_
