/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 11, 2025.
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

#if !defined(__GLIBC__)
#include <fts.h>
#endif

TEST(fts, smoke) {
#if !defined(__GLIBC__)
  char* const paths[] = { const_cast<char*>("."), NULL };
  FTS* fts = fts_open(paths, FTS_PHYSICAL, NULL);
  ASSERT_TRUE(fts != NULL);
  FTSENT* e;
  while ((e = fts_read(fts)) != NULL) {
    ASSERT_EQ(0, fts_set(fts, e, FTS_SKIP));
  }
  ASSERT_EQ(0, fts_close(fts));
#else
  GTEST_SKIP() << "no _FILE_OFFSET_BITS=64 <fts.h> in our old glibc";
#endif
}
