/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 8, 2021.
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
#include "modules/audio_processing/vad/vad_circular_buffer.h"

#include <stdio.h>

#include <memory>

#include "test/gtest.h"

namespace webrtc {

static const int kWidthThreshold = 7;
static const double kValThreshold = 1.0;
static const int kLongBuffSize = 100;
static const int kShortBuffSize = 10;

static void InsertSequentially(int k, VadCircularBuffer* circular_buffer) {
  double mean_val;
  for (int n = 1; n <= k; n++) {
    EXPECT_TRUE(!circular_buffer->is_full());
    circular_buffer->Insert(n);
    mean_val = circular_buffer->Mean();
    EXPECT_EQ((n + 1.0) / 2., mean_val);
  }
}

static void Insert(double value,
                   int num_insertion,
                   VadCircularBuffer* circular_buffer) {
  for (int n = 0; n < num_insertion; n++)
    circular_buffer->Insert(value);
}

static void InsertZeros(int num_zeros, VadCircularBuffer* circular_buffer) {
  Insert(0.0, num_zeros, circular_buffer);
}

TEST(VadCircularBufferTest, GeneralTest) {
  std::unique_ptr<VadCircularBuffer> circular_buffer(
      VadCircularBuffer::Create(kShortBuffSize));
  double mean_val;

  // Mean should return zero if nothing is inserted.
  mean_val = circular_buffer->Mean();
  EXPECT_DOUBLE_EQ(0.0, mean_val);
  InsertSequentially(kShortBuffSize, circular_buffer.get());

  // Should be full.
  EXPECT_TRUE(circular_buffer->is_full());
  // Correct update after being full.
  for (int n = 1; n < kShortBuffSize; n++) {
    circular_buffer->Insert(n);
    mean_val = circular_buffer->Mean();
    EXPECT_DOUBLE_EQ((kShortBuffSize + 1.) / 2., mean_val);
    EXPECT_TRUE(circular_buffer->is_full());
  }

  // Check reset. This should be like starting fresh.
  circular_buffer->Reset();
  mean_val = circular_buffer->Mean();
  EXPECT_DOUBLE_EQ(0, mean_val);
  InsertSequentially(kShortBuffSize, circular_buffer.get());
  EXPECT_TRUE(circular_buffer->is_full());
}

TEST(VadCircularBufferTest, TransientsRemoval) {
  std::unique_ptr<VadCircularBuffer> circular_buffer(
      VadCircularBuffer::Create(kLongBuffSize));
  // Let the first transient be in wrap-around.
  InsertZeros(kLongBuffSize - kWidthThreshold / 2, circular_buffer.get());

  double push_val = kValThreshold;
  double mean_val;
  for (int k = kWidthThreshold; k >= 1; k--) {
    Insert(push_val, k, circular_buffer.get());
    circular_buffer->Insert(0);
    mean_val = circular_buffer->Mean();
    EXPECT_DOUBLE_EQ(k * push_val / kLongBuffSize, mean_val);
    circular_buffer->RemoveTransient(kWidthThreshold, kValThreshold);
    mean_val = circular_buffer->Mean();
    EXPECT_DOUBLE_EQ(0, mean_val);
  }
}

TEST(VadCircularBufferTest, TransientDetection) {
  std::unique_ptr<VadCircularBuffer> circular_buffer(
      VadCircularBuffer::Create(kLongBuffSize));
  // Let the first transient be in wrap-around.
  int num_insertion = kLongBuffSize - kWidthThreshold / 2;
  InsertZeros(num_insertion, circular_buffer.get());

  double push_val = 2;
  // This is longer than a transient and shouldn't be removed.
  int num_non_zero_elements = kWidthThreshold + 1;
  Insert(push_val, num_non_zero_elements, circular_buffer.get());

  double mean_val = circular_buffer->Mean();
  EXPECT_DOUBLE_EQ(num_non_zero_elements * push_val / kLongBuffSize, mean_val);
  circular_buffer->Insert(0);
  EXPECT_EQ(0,
            circular_buffer->RemoveTransient(kWidthThreshold, kValThreshold));
  mean_val = circular_buffer->Mean();
  EXPECT_DOUBLE_EQ(num_non_zero_elements * push_val / kLongBuffSize, mean_val);

  // A transient right after a non-transient, should be removed and mean is
  // not changed.
  num_insertion = 3;
  Insert(push_val, num_insertion, circular_buffer.get());
  circular_buffer->Insert(0);
  EXPECT_EQ(0,
            circular_buffer->RemoveTransient(kWidthThreshold, kValThreshold));
  mean_val = circular_buffer->Mean();
  EXPECT_DOUBLE_EQ(num_non_zero_elements * push_val / kLongBuffSize, mean_val);

  // Last input is larger than threshold, although the sequence is short but
  // it shouldn't be considered transient.
  Insert(push_val, num_insertion, circular_buffer.get());
  num_non_zero_elements += num_insertion;
  EXPECT_EQ(0,
            circular_buffer->RemoveTransient(kWidthThreshold, kValThreshold));
  mean_val = circular_buffer->Mean();
  EXPECT_DOUBLE_EQ(num_non_zero_elements * push_val / kLongBuffSize, mean_val);
}

}  // namespace webrtc
