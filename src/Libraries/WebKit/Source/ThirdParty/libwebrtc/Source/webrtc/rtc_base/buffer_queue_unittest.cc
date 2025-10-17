/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 7, 2022.
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
#include "rtc_base/buffer_queue.h"

#include <string.h>

#include "test/gtest.h"

namespace rtc {

TEST(BufferQueueTest, TestAll) {
  const size_t kSize = 16;
  const char in[kSize * 2 + 1] = "0123456789ABCDEFGHIJKLMNOPQRSTUV";
  char out[kSize * 2];
  size_t bytes;
  BufferQueue queue1(1, kSize);
  BufferQueue queue2(2, kSize);

  // The queue is initially empty.
  EXPECT_EQ(0u, queue1.size());
  EXPECT_FALSE(queue1.ReadFront(out, kSize, &bytes));

  // A write should succeed.
  EXPECT_TRUE(queue1.WriteBack(in, kSize, &bytes));
  EXPECT_EQ(kSize, bytes);
  EXPECT_EQ(1u, queue1.size());

  // The queue is full now (only one buffer allowed).
  EXPECT_FALSE(queue1.WriteBack(in, kSize, &bytes));
  EXPECT_EQ(1u, queue1.size());

  // Reading previously written buffer.
  EXPECT_TRUE(queue1.ReadFront(out, kSize, &bytes));
  EXPECT_EQ(kSize, bytes);
  EXPECT_EQ(0, memcmp(in, out, kSize));

  // The queue is empty again now.
  EXPECT_FALSE(queue1.ReadFront(out, kSize, &bytes));
  EXPECT_EQ(0u, queue1.size());

  // Reading only returns available data.
  EXPECT_TRUE(queue1.WriteBack(in, kSize, &bytes));
  EXPECT_EQ(kSize, bytes);
  EXPECT_EQ(1u, queue1.size());
  EXPECT_TRUE(queue1.ReadFront(out, kSize * 2, &bytes));
  EXPECT_EQ(kSize, bytes);
  EXPECT_EQ(0, memcmp(in, out, kSize));
  EXPECT_EQ(0u, queue1.size());

  // Reading maintains buffer boundaries.
  EXPECT_TRUE(queue2.WriteBack(in, kSize / 2, &bytes));
  EXPECT_EQ(1u, queue2.size());
  EXPECT_TRUE(queue2.WriteBack(in + kSize / 2, kSize / 2, &bytes));
  EXPECT_EQ(2u, queue2.size());
  EXPECT_TRUE(queue2.ReadFront(out, kSize, &bytes));
  EXPECT_EQ(kSize / 2, bytes);
  EXPECT_EQ(0, memcmp(in, out, kSize / 2));
  EXPECT_EQ(1u, queue2.size());
  EXPECT_TRUE(queue2.ReadFront(out, kSize, &bytes));
  EXPECT_EQ(kSize / 2, bytes);
  EXPECT_EQ(0, memcmp(in + kSize / 2, out, kSize / 2));
  EXPECT_EQ(0u, queue2.size());

  // Reading truncates buffers.
  EXPECT_TRUE(queue2.WriteBack(in, kSize / 2, &bytes));
  EXPECT_EQ(1u, queue2.size());
  EXPECT_TRUE(queue2.WriteBack(in + kSize / 2, kSize / 2, &bytes));
  EXPECT_EQ(2u, queue2.size());
  // Read first packet partially in too-small buffer.
  EXPECT_TRUE(queue2.ReadFront(out, kSize / 4, &bytes));
  EXPECT_EQ(kSize / 4, bytes);
  EXPECT_EQ(0, memcmp(in, out, kSize / 4));
  EXPECT_EQ(1u, queue2.size());
  // Remainder of first packet is truncated, reading starts with next packet.
  EXPECT_TRUE(queue2.ReadFront(out, kSize, &bytes));
  EXPECT_EQ(kSize / 2, bytes);
  EXPECT_EQ(0, memcmp(in + kSize / 2, out, kSize / 2));
  EXPECT_EQ(0u, queue2.size());
}

}  // namespace rtc
