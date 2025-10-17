/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 17, 2024.
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
#include "rtc_base/zero_memory.h"

#include <stdint.h>

#include "api/array_view.h"
#include "test/gtest.h"

namespace rtc {

TEST(ZeroMemoryTest, TestZeroMemory) {
  static const size_t kBufferSize = 32;
  uint8_t buffer[kBufferSize];
  for (size_t i = 0; i < kBufferSize; i++) {
    buffer[i] = static_cast<uint8_t>(i + 1);
  }
  ExplicitZeroMemory(buffer, sizeof(buffer));
  for (size_t i = 0; i < kBufferSize; i++) {
    EXPECT_EQ(buffer[i], 0);
  }
}

TEST(ZeroMemoryTest, TestZeroArrayView) {
  static const size_t kBufferSize = 32;
  uint8_t buffer[kBufferSize];
  for (size_t i = 0; i < kBufferSize; i++) {
    buffer[i] = static_cast<uint8_t>(i + 1);
  }
  ExplicitZeroMemory(rtc::ArrayView<uint8_t>(buffer, sizeof(buffer)));
  for (size_t i = 0; i < kBufferSize; i++) {
    EXPECT_EQ(buffer[i], 0);
  }
}

// While this test doesn't actually test anything, it can be used to check
// the compiler output to make sure the call to "ExplicitZeroMemory" is not
// optimized away.
TEST(ZeroMemoryTest, TestZeroMemoryUnused) {
  static const size_t kBufferSize = 32;
  uint8_t buffer[kBufferSize];
  ExplicitZeroMemory(buffer, sizeof(buffer));
}

}  // namespace rtc
