/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 31, 2023.
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

#include <sys/io.h>

#include "utils.h"

TEST(sys_io, iopl) {
#if defined(__i386__) || defined(__x86_64__)
  errno = 0;
  ASSERT_EQ(-1, iopl(4));
  ASSERT_ERRNO(EINVAL);
#else
  GTEST_SKIP() << "iopl requires x86/x86-64";
#endif
}

TEST(sys_io, ioperm) {
#if defined(__i386__) || defined(__x86_64__)
  errno = 0;
  ASSERT_EQ(-1, ioperm(65535, 4, 0));
  ASSERT_ERRNO(EINVAL);
#else
  GTEST_SKIP() << "ioperm requires x86/x86-64";
#endif
}
