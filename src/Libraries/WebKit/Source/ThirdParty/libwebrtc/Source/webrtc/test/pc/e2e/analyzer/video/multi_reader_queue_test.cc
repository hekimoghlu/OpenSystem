/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 18, 2023.
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
#include "test/pc/e2e/analyzer/video/multi_reader_queue.h"

#include <optional>

#include "test/gtest.h"

namespace webrtc {
namespace {

TEST(MultiReaderQueueTest, EmptyQueueEmptyForAllHeads) {
  MultiReaderQueue<int> queue = MultiReaderQueue<int>(/*readers_count=*/10);
  EXPECT_EQ(queue.size(), 0lu);
  for (int i = 0; i < 10; ++i) {
    EXPECT_TRUE(queue.IsEmpty(/*reader=*/i));
    EXPECT_EQ(queue.size(/*reader=*/i), 0lu);
    EXPECT_FALSE(queue.PopFront(/*reader=*/i).has_value());
    EXPECT_FALSE(queue.Front(/*reader=*/i).has_value());
  }
}

TEST(MultiReaderQueueTest, SizeIsEqualForAllHeadsAfterAddOnly) {
  MultiReaderQueue<int> queue = MultiReaderQueue<int>(/*readers_count=*/10);
  queue.PushBack(1);
  queue.PushBack(2);
  queue.PushBack(3);
  EXPECT_EQ(queue.size(), 3lu);
  for (int i = 0; i < 10; ++i) {
    EXPECT_FALSE(queue.IsEmpty(/*reader=*/i));
    EXPECT_EQ(queue.size(/*reader=*/i), 3lu);
  }
}

TEST(MultiReaderQueueTest, SizeIsCorrectAfterRemoveFromOnlyOneHead) {
  MultiReaderQueue<int> queue = MultiReaderQueue<int>(/*readers_count=*/10);
  for (int i = 0; i < 5; ++i) {
    queue.PushBack(i);
  }
  EXPECT_EQ(queue.size(), 5lu);
  // Removing elements from queue #0
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(queue.size(/*reader=*/0), static_cast<size_t>(5 - i));
    EXPECT_EQ(queue.PopFront(/*reader=*/0), std::optional<int>(i));
    for (int j = 1; j < 10; ++j) {
      EXPECT_EQ(queue.size(/*reader=*/j), 5lu);
    }
  }
  EXPECT_EQ(queue.size(/*reader=*/0), 0lu);
  EXPECT_TRUE(queue.IsEmpty(/*reader=*/0));
}

TEST(MultiReaderQueueTest, SingleHeadOneAddOneRemove) {
  MultiReaderQueue<int> queue = MultiReaderQueue<int>(/*readers_count=*/1);
  queue.PushBack(1);
  EXPECT_EQ(queue.size(), 1lu);
  EXPECT_TRUE(queue.Front(/*reader=*/0).has_value());
  EXPECT_EQ(queue.Front(/*reader=*/0).value(), 1);
  std::optional<int> value = queue.PopFront(/*reader=*/0);
  EXPECT_TRUE(value.has_value());
  EXPECT_EQ(value.value(), 1);
  EXPECT_EQ(queue.size(), 0lu);
  EXPECT_TRUE(queue.IsEmpty(/*reader=*/0));
}

TEST(MultiReaderQueueTest, SingleHead) {
  MultiReaderQueue<size_t> queue =
      MultiReaderQueue<size_t>(/*readers_count=*/1);
  for (size_t i = 0; i < 10; ++i) {
    queue.PushBack(i);
    EXPECT_EQ(queue.size(), i + 1);
  }
  for (size_t i = 0; i < 10; ++i) {
    EXPECT_EQ(queue.Front(/*reader=*/0), std::optional<size_t>(i));
    EXPECT_EQ(queue.PopFront(/*reader=*/0), std::optional<size_t>(i));
    EXPECT_EQ(queue.size(), 10 - i - 1);
  }
}

TEST(MultiReaderQueueTest, ThreeHeadsAddAllRemoveAllPerHead) {
  MultiReaderQueue<size_t> queue =
      MultiReaderQueue<size_t>(/*readers_count=*/3);
  for (size_t i = 0; i < 10; ++i) {
    queue.PushBack(i);
    EXPECT_EQ(queue.size(), i + 1);
  }
  for (size_t i = 0; i < 10; ++i) {
    std::optional<size_t> value = queue.PopFront(/*reader=*/0);
    EXPECT_EQ(queue.size(), 10lu);
    ASSERT_TRUE(value.has_value());
    EXPECT_EQ(value.value(), i);
  }
  for (size_t i = 0; i < 10; ++i) {
    std::optional<size_t> value = queue.PopFront(/*reader=*/1);
    EXPECT_EQ(queue.size(), 10lu);
    ASSERT_TRUE(value.has_value());
    EXPECT_EQ(value.value(), i);
  }
  for (size_t i = 0; i < 10; ++i) {
    std::optional<size_t> value = queue.PopFront(/*reader=*/2);
    EXPECT_EQ(queue.size(), 10 - i - 1);
    ASSERT_TRUE(value.has_value());
    EXPECT_EQ(value.value(), i);
  }
}

TEST(MultiReaderQueueTest, ThreeHeadsAddAllRemoveAll) {
  MultiReaderQueue<size_t> queue =
      MultiReaderQueue<size_t>(/*readers_count=*/3);
  for (size_t i = 0; i < 10; ++i) {
    queue.PushBack(i);
    EXPECT_EQ(queue.size(), i + 1);
  }
  for (size_t i = 0; i < 10; ++i) {
    std::optional<size_t> value1 = queue.PopFront(/*reader=*/0);
    std::optional<size_t> value2 = queue.PopFront(/*reader=*/1);
    std::optional<size_t> value3 = queue.PopFront(/*reader=*/2);
    EXPECT_EQ(queue.size(), 10 - i - 1);
    ASSERT_TRUE(value1.has_value());
    ASSERT_TRUE(value2.has_value());
    ASSERT_TRUE(value3.has_value());
    EXPECT_EQ(value1.value(), i);
    EXPECT_EQ(value2.value(), i);
    EXPECT_EQ(value3.value(), i);
  }
}

TEST(MultiReaderQueueTest, AddReaderSeeElementsOnlyFromReaderToCopy) {
  MultiReaderQueue<size_t> queue =
      MultiReaderQueue<size_t>(/*readers_count=*/2);
  for (size_t i = 0; i < 10; ++i) {
    queue.PushBack(i);
  }
  for (size_t i = 0; i < 5; ++i) {
    queue.PopFront(0);
  }

  queue.AddReader(/*reader=*/2, /*reader_to_copy=*/0);

  EXPECT_EQ(queue.readers_count(), 3lu);
  for (size_t i = 5; i < 10; ++i) {
    std::optional<size_t> value = queue.PopFront(/*reader=*/2);
    EXPECT_EQ(queue.size(/*reader=*/2), 10 - i - 1);
    ASSERT_TRUE(value.has_value());
    EXPECT_EQ(value.value(), i);
  }
}

TEST(MultiReaderQueueTest, AddReaderWithoutReaderToCopySeeFullQueue) {
  MultiReaderQueue<size_t> queue =
      MultiReaderQueue<size_t>(/*readers_count=*/2);
  for (size_t i = 0; i < 10; ++i) {
    queue.PushBack(i);
  }
  for (size_t i = 0; i < 5; ++i) {
    queue.PopFront(/*reader=*/0);
  }

  queue.AddReader(/*reader=*/2);

  EXPECT_EQ(queue.readers_count(), 3lu);
  for (size_t i = 0; i < 10; ++i) {
    std::optional<size_t> value = queue.PopFront(/*reader=*/2);
    EXPECT_EQ(queue.size(/*reader=*/2), 10 - i - 1);
    ASSERT_TRUE(value.has_value());
    EXPECT_EQ(value.value(), i);
  }
}

TEST(MultiReaderQueueTest, RemoveReaderWontChangeOthers) {
  MultiReaderQueue<size_t> queue =
      MultiReaderQueue<size_t>(/*readers_count=*/2);
  for (size_t i = 0; i < 10; ++i) {
    queue.PushBack(i);
  }
  EXPECT_EQ(queue.size(/*reader=*/1), 10lu);

  queue.RemoveReader(0);

  EXPECT_EQ(queue.readers_count(), 1lu);
  EXPECT_EQ(queue.size(/*reader=*/1), 10lu);
}

TEST(MultiReaderQueueTest, RemoveLastReaderMakesQueueEmpty) {
  MultiReaderQueue<size_t> queue =
      MultiReaderQueue<size_t>(/*readers_count=*/1);
  for (size_t i = 0; i < 10; ++i) {
    queue.PushBack(i);
  }
  EXPECT_EQ(queue.size(), 10lu);

  queue.RemoveReader(0);

  EXPECT_EQ(queue.size(), 0lu);
  EXPECT_EQ(queue.readers_count(), 0lu);
}

}  // namespace
}  // namespace webrtc
