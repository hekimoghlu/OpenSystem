/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 29, 2024.
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
#include <gtest/gtest.h>

#include "linker_gnu_hash.h"

TEST(linker_gnu_hash, compare_neon_to_simple) {
#if USE_GNU_HASH_NEON
  auto check_input = [&](const char* name) {
    auto expected = calculate_gnu_hash_simple(name);
    auto actual = calculate_gnu_hash_neon(name);
    EXPECT_EQ(expected.first, actual.first) << name;
    EXPECT_EQ(expected.second, actual.second) << name;
  };

  __attribute__((aligned(8))) const char test1[] = "abcdefghijklmnop\0qrstuvwxyz";
  for (size_t i = 0; i < sizeof(test1) - 1; ++i) {
    check_input(&test1[i]);
  }

  __attribute__((aligned(8))) const char test2[] = "abcdefghijklmnopqrs\0tuvwxyz";
  for (size_t i = 0; i < sizeof(test2) - 1; ++i) {
    check_input(&test2[i]);
  }

  __attribute__((aligned(8))) const char test3[] = "abcdefghijklmnopqrstuv\0wxyz";
  for (size_t i = 0; i < sizeof(test3) - 1; ++i) {
    check_input(&test3[i]);
  }
#else
  GTEST_SKIP() << "This test is only implemented on arm/arm64";
#endif
}
