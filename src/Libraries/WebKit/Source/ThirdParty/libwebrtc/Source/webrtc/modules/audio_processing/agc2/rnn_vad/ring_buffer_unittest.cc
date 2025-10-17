/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 3, 2025.
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
#include "modules/audio_processing/agc2/rnn_vad/ring_buffer.h"

#include "test/gtest.h"

namespace webrtc {
namespace rnn_vad {
namespace {

// Compare the elements of two given array views.
template <typename T, std::ptrdiff_t S>
void ExpectEq(rtc::ArrayView<const T, S> a, rtc::ArrayView<const T, S> b) {
  for (int i = 0; i < S; ++i) {
    SCOPED_TRACE(i);
    EXPECT_EQ(a[i], b[i]);
  }
}

// Test push/read sequences.
template <typename T, int S, int N>
void TestRingBuffer() {
  SCOPED_TRACE(N);
  SCOPED_TRACE(S);
  std::array<T, S> prev_pushed_array;
  std::array<T, S> pushed_array;
  rtc::ArrayView<const T, S> pushed_array_view(pushed_array.data(), S);

  // Init.
  RingBuffer<T, S, N> ring_buf;
  ring_buf.GetArrayView(0);
  pushed_array.fill(0);
  ring_buf.Push(pushed_array_view);
  ExpectEq(pushed_array_view, ring_buf.GetArrayView(0));

  // Push N times and check most recent and second most recent.
  for (T v = 1; v <= static_cast<T>(N); ++v) {
    SCOPED_TRACE(v);
    prev_pushed_array = pushed_array;
    pushed_array.fill(v);
    ring_buf.Push(pushed_array_view);
    ExpectEq(pushed_array_view, ring_buf.GetArrayView(0));
    if (N > 1) {
      pushed_array.fill(v - 1);
      ExpectEq(pushed_array_view, ring_buf.GetArrayView(1));
    }
  }

  // Check buffer.
  for (int delay = 2; delay < N; ++delay) {
    SCOPED_TRACE(delay);
    T expected_value = N - static_cast<T>(delay);
    pushed_array.fill(expected_value);
    ExpectEq(pushed_array_view, ring_buf.GetArrayView(delay));
  }
}

// Check that for different delays, different views are returned.
TEST(RnnVadTest, RingBufferArrayViews) {
  constexpr int s = 3;
  constexpr int n = 4;
  RingBuffer<int, s, n> ring_buf;
  std::array<int, s> pushed_array;
  pushed_array.fill(1);
  for (int k = 0; k <= n; ++k) {  // Push data n + 1 times.
    SCOPED_TRACE(k);
    // Check array views.
    for (int i = 0; i < n; ++i) {
      SCOPED_TRACE(i);
      auto view_i = ring_buf.GetArrayView(i);
      for (int j = i + 1; j < n; ++j) {
        SCOPED_TRACE(j);
        auto view_j = ring_buf.GetArrayView(j);
        EXPECT_NE(view_i, view_j);
      }
    }
    ring_buf.Push(pushed_array);
  }
}

TEST(RnnVadTest, RingBufferUnsigned) {
  TestRingBuffer<uint8_t, 1, 1>();
  TestRingBuffer<uint8_t, 2, 5>();
  TestRingBuffer<uint8_t, 5, 2>();
  TestRingBuffer<uint8_t, 5, 5>();
}

TEST(RnnVadTest, RingBufferSigned) {
  TestRingBuffer<int, 1, 1>();
  TestRingBuffer<int, 2, 5>();
  TestRingBuffer<int, 5, 2>();
  TestRingBuffer<int, 5, 5>();
}

TEST(RnnVadTest, RingBufferFloating) {
  TestRingBuffer<float, 1, 1>();
  TestRingBuffer<float, 2, 5>();
  TestRingBuffer<float, 5, 2>();
  TestRingBuffer<float, 5, 5>();
}

}  // namespace
}  // namespace rnn_vad
}  // namespace webrtc
