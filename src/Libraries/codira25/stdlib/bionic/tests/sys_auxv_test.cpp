/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 27, 2022.
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

#include <sys/auxv.h>

TEST(sys_auxv, getauxval_HWCAP) {
  __attribute__((__unused__)) unsigned long hwcap = getauxval(AT_HWCAP);

  // Check that the constants for *using* AT_HWCAP are also available.
#if defined(__arm__)
  ASSERT_NE(0, HWCAP_THUMB);
#elif defined(__aarch64__)
  ASSERT_NE(0, HWCAP_FP);
#endif
}

TEST(sys_auxv, getauxval_HWCAP2) {
#if defined(AT_HWCAP2)
  __attribute__((__unused__)) unsigned long hwcap = getauxval(AT_HWCAP2);

  // Check that the constants for *using* AT_HWCAP2 are also available.
#if defined(__arm__)
  ASSERT_NE(0, HWCAP2_AES);
#elif defined(__aarch64__)
  ASSERT_NE(0, HWCAP2_SVE2);
#endif
#else
  GTEST_SKIP() << "No AT_HWCAP2 for this architecture.";
#endif
}
