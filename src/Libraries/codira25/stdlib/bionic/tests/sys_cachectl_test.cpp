/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 18, 2025.
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

#if __has_include(<sys/cachectl.h>)
#include <sys/cachectl.h>
#endif

TEST(sys_cachectl, __riscv_flush_icache) {
#if defined(__riscv) && __has_include(<sys/cachectl.h>)
  // As of Linux 6.4, the address range is ignored (so nullptr is fine),
  // and the flags you actually want are 0 ("all threads"),
  // but we test the unusual flag just to make sure it works.
  ASSERT_EQ(0, __riscv_flush_icache(nullptr, nullptr, SYS_RISCV_FLUSH_ICACHE_LOCAL));
#else
  GTEST_SKIP() << "__riscv_flush_icache requires riscv64";
#endif
}
