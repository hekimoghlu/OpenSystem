/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 6, 2023.
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
#include "modules/audio_processing/agc2/rnn_vad/sequence_buffer.h"

#include <algorithm>
#include <array>

#include "test/gtest.h"

namespace webrtc {
namespace rnn_vad {
namespace {

template <typename T, int S, int N>
void TestSequenceBufferPushOp() {
  SCOPED_TRACE(S);
  SCOPED_TRACE(N);
  SequenceBuffer<T, S, N> seq_buf;
  auto seq_buf_view = seq_buf.GetBufferView();
  std::array<T, N> chunk;

  // Check that a chunk is fully gone after ceil(S / N) push ops.
  chunk.fill(1);
  seq_buf.Push(chunk);
  chunk.fill(0);
  constexpr int required_push_ops = (S % N) ? S / N + 1 : S / N;
  for (int i = 0; i < required_push_ops - 1; ++i) {
    SCOPED_TRACE(i);
    seq_buf.Push(chunk);
    // Still in the buffer.
    const auto* m = std::max_element(seq_buf_view.begin(), seq_buf_view.end());
    EXPECT_EQ(1, *m);
  }
  // Gone after another push.
  seq_buf.Push(chunk);
  const auto* m = std::max_element(seq_buf_view.begin(), seq_buf_view.end());
  EXPECT_EQ(0, *m);

  // Check that the last item moves left by N positions after a push op.
  if (S > N) {
    // Fill in with non-zero values.
    for (int i = 0; i < N; ++i)
      chunk[i] = static_cast<T>(i + 1);
    seq_buf.Push(chunk);
    // With the next Push(), `last` will be moved left by N positions.
    const T last = chunk[N - 1];
    for (int i = 0; i < N; ++i)
      chunk[i] = static_cast<T>(last + i + 1);
    seq_buf.Push(chunk);
    EXPECT_EQ(last, seq_buf_view[S - N - 1]);
  }
}

TEST(RnnVadTest, SequenceBufferGetters) {
  constexpr int buffer_size = 8;
  constexpr int chunk_size = 8;
  SequenceBuffer<int, buffer_size, chunk_size> seq_buf;
  EXPECT_EQ(buffer_size, seq_buf.size());
  EXPECT_EQ(chunk_size, seq_buf.chunks_size());
  // Test view.
  auto seq_buf_view = seq_buf.GetBufferView();
  EXPECT_EQ(0, seq_buf_view[0]);
  EXPECT_EQ(0, seq_buf_view[seq_buf_view.size() - 1]);
  constexpr std::array<int, chunk_size> chunk = {10, 20, 30, 40,
                                                 50, 60, 70, 80};
  seq_buf.Push(chunk);
  EXPECT_EQ(10, *seq_buf_view.begin());
  EXPECT_EQ(80, *(seq_buf_view.end() - 1));
}

TEST(RnnVadTest, SequenceBufferPushOpsUnsigned) {
  TestSequenceBufferPushOp<uint8_t, 32, 8>();   // Chunk size: 25%.
  TestSequenceBufferPushOp<uint8_t, 32, 16>();  // Chunk size: 50%.
  TestSequenceBufferPushOp<uint8_t, 32, 32>();  // Chunk size: 100%.
  TestSequenceBufferPushOp<uint8_t, 23, 7>();   // Non-integer ratio.
}

TEST(RnnVadTest, SequenceBufferPushOpsSigned) {
  TestSequenceBufferPushOp<int, 32, 8>();   // Chunk size: 25%.
  TestSequenceBufferPushOp<int, 32, 16>();  // Chunk size: 50%.
  TestSequenceBufferPushOp<int, 32, 32>();  // Chunk size: 100%.
  TestSequenceBufferPushOp<int, 23, 7>();   // Non-integer ratio.
}

TEST(RnnVadTest, SequenceBufferPushOpsFloating) {
  TestSequenceBufferPushOp<float, 32, 8>();   // Chunk size: 25%.
  TestSequenceBufferPushOp<float, 32, 16>();  // Chunk size: 50%.
  TestSequenceBufferPushOp<float, 32, 32>();  // Chunk size: 100%.
  TestSequenceBufferPushOp<float, 23, 7>();   // Non-integer ratio.
}

}  // namespace
}  // namespace rnn_vad
}  // namespace webrtc
