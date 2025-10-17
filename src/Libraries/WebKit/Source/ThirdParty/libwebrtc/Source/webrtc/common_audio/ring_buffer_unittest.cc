/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 14, 2024.
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
#include "common_audio/ring_buffer.h"

#include <stdlib.h>
#include <time.h>

#include <algorithm>
#include <memory>

#include "test/gtest.h"

namespace webrtc {

struct FreeBufferDeleter {
  inline void operator()(void* ptr) const { WebRtc_FreeBuffer(ptr); }
};
typedef std::unique_ptr<RingBuffer, FreeBufferDeleter> scoped_ring_buffer;

static void AssertElementEq(int expected, int actual) {
  ASSERT_EQ(expected, actual);
}

static int SetIncrementingData(int* data,
                               int num_elements,
                               int starting_value) {
  for (int i = 0; i < num_elements; i++) {
    data[i] = starting_value++;
  }
  return starting_value;
}

static int CheckIncrementingData(int* data,
                                 int num_elements,
                                 int starting_value) {
  for (int i = 0; i < num_elements; i++) {
    AssertElementEq(starting_value++, data[i]);
  }
  return starting_value;
}

// We use ASSERTs in this test to avoid obscuring the seed in the case of a
// failure.
static void RandomStressTest(int** data_ptr) {
  const int kNumTests = 10;
  const int kNumOps = 1000;
  const int kMaxBufferSize = 1000;

  unsigned int seed = time(nullptr);
  printf("seed=%u\n", seed);
  srand(seed);
  for (int i = 0; i < kNumTests; i++) {
    // rand_r is not supported on many platforms, so rand is used.
    const int buffer_size = std::max(rand() % kMaxBufferSize, 1);  // NOLINT
    std::unique_ptr<int[]> write_data(new int[buffer_size]);
    std::unique_ptr<int[]> read_data(new int[buffer_size]);
    scoped_ring_buffer buffer(WebRtc_CreateBuffer(buffer_size, sizeof(int)));
    ASSERT_TRUE(buffer.get() != nullptr);
    WebRtc_InitBuffer(buffer.get());
    int buffer_consumed = 0;
    int write_element = 0;
    int read_element = 0;
    for (int j = 0; j < kNumOps; j++) {
      const bool write = rand() % 2 == 0 ? true : false;  // NOLINT
      const int num_elements = rand() % buffer_size;      // NOLINT
      if (write) {
        const int buffer_available = buffer_size - buffer_consumed;
        ASSERT_EQ(static_cast<size_t>(buffer_available),
                  WebRtc_available_write(buffer.get()));
        const int expected_elements = std::min(num_elements, buffer_available);
        write_element = SetIncrementingData(write_data.get(), expected_elements,
                                            write_element);
        ASSERT_EQ(
            static_cast<size_t>(expected_elements),
            WebRtc_WriteBuffer(buffer.get(), write_data.get(), num_elements));
        buffer_consumed =
            std::min(buffer_consumed + expected_elements, buffer_size);
      } else {
        const int expected_elements = std::min(num_elements, buffer_consumed);
        ASSERT_EQ(static_cast<size_t>(buffer_consumed),
                  WebRtc_available_read(buffer.get()));
        ASSERT_EQ(
            static_cast<size_t>(expected_elements),
            WebRtc_ReadBuffer(buffer.get(), reinterpret_cast<void**>(data_ptr),
                              read_data.get(), num_elements));
        int* check_ptr = read_data.get();
        if (data_ptr) {
          check_ptr = *data_ptr;
        }
        read_element =
            CheckIncrementingData(check_ptr, expected_elements, read_element);
        buffer_consumed = std::max(buffer_consumed - expected_elements, 0);
      }
    }
  }
}

TEST(RingBufferTest, RandomStressTest) {
  int* data_ptr = nullptr;
  RandomStressTest(&data_ptr);
}

TEST(RingBufferTest, RandomStressTestWithNullPtr) {
  RandomStressTest(nullptr);
}

TEST(RingBufferTest, PassingNulltoReadBufferForcesMemcpy) {
  const size_t kDataSize = 2;
  int write_data[kDataSize];
  int read_data[kDataSize];
  int* data_ptr;

  scoped_ring_buffer buffer(WebRtc_CreateBuffer(kDataSize, sizeof(int)));
  ASSERT_TRUE(buffer.get() != nullptr);
  WebRtc_InitBuffer(buffer.get());

  SetIncrementingData(write_data, kDataSize, 0);
  EXPECT_EQ(kDataSize, WebRtc_WriteBuffer(buffer.get(), write_data, kDataSize));
  SetIncrementingData(read_data, kDataSize, kDataSize);
  EXPECT_EQ(kDataSize,
            WebRtc_ReadBuffer(buffer.get(), reinterpret_cast<void**>(&data_ptr),
                              read_data, kDataSize));
  // Copying was not necessary, so `read_data` has not been updated.
  CheckIncrementingData(data_ptr, kDataSize, 0);
  CheckIncrementingData(read_data, kDataSize, kDataSize);

  EXPECT_EQ(kDataSize, WebRtc_WriteBuffer(buffer.get(), write_data, kDataSize));
  EXPECT_EQ(kDataSize,
            WebRtc_ReadBuffer(buffer.get(), nullptr, read_data, kDataSize));
  // Passing null forces a memcpy, so `read_data` is now updated.
  CheckIncrementingData(read_data, kDataSize, 0);
}

TEST(RingBufferTest, CreateHandlesErrors) {
  EXPECT_TRUE(WebRtc_CreateBuffer(0, 1) == nullptr);
  EXPECT_TRUE(WebRtc_CreateBuffer(1, 0) == nullptr);
  RingBuffer* buffer = WebRtc_CreateBuffer(1, 1);
  EXPECT_TRUE(buffer != nullptr);
  WebRtc_FreeBuffer(buffer);
}

}  // namespace webrtc
