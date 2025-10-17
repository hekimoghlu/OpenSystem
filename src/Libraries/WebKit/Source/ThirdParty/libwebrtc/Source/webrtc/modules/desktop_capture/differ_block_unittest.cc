/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 3, 2025.
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
#include "modules/desktop_capture/differ_block.h"

#include <string.h>

#include "test/gtest.h"

namespace webrtc {

// Run 900 times to mimic 1280x720.
// TODO(fbarchard): Remove benchmark once performance is non-issue.
static const int kTimesToRun = 900;

static void GenerateData(uint8_t* data, int size) {
  for (int i = 0; i < size; ++i) {
    data[i] = i;
  }
}

// Memory buffer large enough for 2 blocks aligned to 16 bytes.
static const int kSizeOfBlock = kBlockSize * kBlockSize * kBytesPerPixel;
uint8_t block_buffer[kSizeOfBlock * 2 + 16];

void PrepareBuffers(uint8_t*& block1, uint8_t*& block2) {
  block1 = reinterpret_cast<uint8_t*>(
      (reinterpret_cast<uintptr_t>(&block_buffer[0]) + 15) & ~15);
  GenerateData(block1, kSizeOfBlock);
  block2 = block1 + kSizeOfBlock;
  memcpy(block2, block1, kSizeOfBlock);
}

TEST(BlockDifferenceTestSame, BlockDifference) {
  uint8_t* block1;
  uint8_t* block2;
  PrepareBuffers(block1, block2);

  // These blocks should match.
  for (int i = 0; i < kTimesToRun; ++i) {
    int result = BlockDifference(block1, block2, kBlockSize * kBytesPerPixel);
    EXPECT_EQ(0, result);
  }
}

TEST(BlockDifferenceTestLast, BlockDifference) {
  uint8_t* block1;
  uint8_t* block2;
  PrepareBuffers(block1, block2);
  block2[kSizeOfBlock - 2] += 1;

  for (int i = 0; i < kTimesToRun; ++i) {
    int result = BlockDifference(block1, block2, kBlockSize * kBytesPerPixel);
    EXPECT_EQ(1, result);
  }
}

TEST(BlockDifferenceTestMid, BlockDifference) {
  uint8_t* block1;
  uint8_t* block2;
  PrepareBuffers(block1, block2);
  block2[kSizeOfBlock / 2 + 1] += 1;

  for (int i = 0; i < kTimesToRun; ++i) {
    int result = BlockDifference(block1, block2, kBlockSize * kBytesPerPixel);
    EXPECT_EQ(1, result);
  }
}

TEST(BlockDifferenceTestFirst, BlockDifference) {
  uint8_t* block1;
  uint8_t* block2;
  PrepareBuffers(block1, block2);
  block2[0] += 1;

  for (int i = 0; i < kTimesToRun; ++i) {
    int result = BlockDifference(block1, block2, kBlockSize * kBytesPerPixel);
    EXPECT_EQ(1, result);
  }
}

}  // namespace webrtc
