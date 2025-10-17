/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 22, 2025.
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
#include "rtc_base/memory/aligned_malloc.h"

#include <memory>

#ifdef _WIN32
#include <windows.h>
#else
#include <stdint.h>
#endif

#include "test/gtest.h"

namespace webrtc {

// Returns true if `size` and `alignment` are valid combinations.
bool CorrectUsage(size_t size, size_t alignment) {
  std::unique_ptr<char, AlignedFreeDeleter> scoped(
      static_cast<char*>(AlignedMalloc(size, alignment)));
  if (scoped.get() == NULL) {
    return false;
  }
  const uintptr_t scoped_address = reinterpret_cast<uintptr_t>(scoped.get());
  return 0u == scoped_address % alignment;
}

TEST(AlignedMalloc, GetRightAlign) {
  const size_t size = 100;
  const size_t alignment = 32;
  const size_t left_misalignment = 1;
  std::unique_ptr<char, AlignedFreeDeleter> scoped(
      static_cast<char*>(AlignedMalloc(size, alignment)));
  EXPECT_TRUE(scoped.get() != NULL);
  const uintptr_t aligned_address = reinterpret_cast<uintptr_t>(scoped.get());
  const uintptr_t misaligned_address = aligned_address - left_misalignment;
  const char* misaligned_ptr =
      reinterpret_cast<const char*>(misaligned_address);
  const char* realigned_ptr = GetRightAlign(misaligned_ptr, alignment);
  EXPECT_EQ(scoped.get(), realigned_ptr);
}

TEST(AlignedMalloc, IncorrectSize) {
  const size_t incorrect_size = 0;
  const size_t alignment = 64;
  EXPECT_FALSE(CorrectUsage(incorrect_size, alignment));
}

TEST(AlignedMalloc, IncorrectAlignment) {
  const size_t size = 100;
  const size_t incorrect_alignment = 63;
  EXPECT_FALSE(CorrectUsage(size, incorrect_alignment));
}

TEST(AlignedMalloc, AlignTo2Bytes) {
  size_t size = 100;
  size_t alignment = 2;
  EXPECT_TRUE(CorrectUsage(size, alignment));
}

TEST(AlignedMalloc, AlignTo32Bytes) {
  size_t size = 100;
  size_t alignment = 32;
  EXPECT_TRUE(CorrectUsage(size, alignment));
}

TEST(AlignedMalloc, AlignTo128Bytes) {
  size_t size = 100;
  size_t alignment = 128;
  EXPECT_TRUE(CorrectUsage(size, alignment));
}

}  // namespace webrtc
