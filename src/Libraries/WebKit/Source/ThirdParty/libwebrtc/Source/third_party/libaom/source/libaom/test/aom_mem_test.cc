/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 18, 2023.
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
#include "aom_mem/aom_mem.h"

#include <cstdio>
#include <cstddef>

#include "gtest/gtest.h"

TEST(AomMemTest, Overflow) {
  // Allocations are aligned > 1 so SIZE_MAX should always fail.
  ASSERT_EQ(aom_malloc(SIZE_MAX), nullptr);
  ASSERT_EQ(aom_calloc(1, SIZE_MAX), nullptr);
  ASSERT_EQ(aom_calloc(32, SIZE_MAX / 32), nullptr);
  ASSERT_EQ(aom_calloc(SIZE_MAX, SIZE_MAX), nullptr);
  ASSERT_EQ(aom_memalign(1, SIZE_MAX), nullptr);
  ASSERT_EQ(aom_memalign(64, SIZE_MAX), nullptr);
  ASSERT_EQ(aom_memalign(64, SIZE_MAX - 64), nullptr);
  ASSERT_EQ(aom_memalign(64, SIZE_MAX - 64 - sizeof(size_t) + 2), nullptr);
}

TEST(AomMemTest, NullParams) {
  ASSERT_EQ(aom_memset16(nullptr, 0, 0), nullptr);
  aom_free(nullptr);
}
