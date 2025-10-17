/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 29, 2022.
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

#include <fnmatch.h>

TEST(fnmatch, basic) {
  EXPECT_EQ(0, fnmatch("abc", "abc", 0));
  EXPECT_EQ(FNM_NOMATCH, fnmatch("abc", "abd", 0));
}

TEST(fnmatch, casefold) {
  EXPECT_EQ(FNM_NOMATCH, fnmatch("abc", "aBc", 0));
  EXPECT_EQ(0, fnmatch("abc", "aBc", FNM_CASEFOLD));
}

TEST(fnmatch, character_class) {
  // Literal.
  EXPECT_EQ(0, fnmatch("ab[cd]", "abc", 0));
  EXPECT_EQ(0, fnmatch("ab[cd]", "abd", 0));
  EXPECT_EQ(FNM_NOMATCH, fnmatch("ab[cd]", "abe", 0));

  // Inverted literal.
  EXPECT_EQ(FNM_NOMATCH, fnmatch("ab[^cd]", "abc", 0));
  EXPECT_EQ(FNM_NOMATCH, fnmatch("ab[^cd]", "abd", 0));
  EXPECT_EQ(0, fnmatch("ab[^cd]", "abe", 0));

  // Range.
  EXPECT_EQ(0, fnmatch("a[0-9]b", "a0b", 0));
  EXPECT_EQ(FNM_NOMATCH, fnmatch("a[0-9]b", "aOb", 0));

  // Inverted range.
  EXPECT_EQ(0, fnmatch("a[^0-9]b", "aOb", 0));
  EXPECT_EQ(FNM_NOMATCH, fnmatch("a[^0-9]b", "a0b", 0));

  // Named.
  EXPECT_EQ(0, fnmatch("a[[:digit:]]b", "a0b", 0));
  EXPECT_EQ(FNM_NOMATCH, fnmatch("a[[:digit:]]b", "aOb", 0));

  // Inverted named.
  EXPECT_EQ(FNM_NOMATCH, fnmatch("a[^[:digit:]]b", "a0b", 0));
  EXPECT_EQ(0, fnmatch("a[^[:digit:]]b", "aOb", 0));
}

TEST(fnmatch, wild_any) {
  EXPECT_EQ(0, fnmatch("ab*", "ab", 0));
  EXPECT_EQ(0, fnmatch("ab*", "abc", 0));
  EXPECT_EQ(0, fnmatch("ab*", "abcd", 0));
  EXPECT_EQ(0, fnmatch("ab*", "ab/cd", 0));
  EXPECT_EQ(FNM_NOMATCH, fnmatch("ab*", "ab/cd", FNM_PATHNAME));
}

TEST(fnmatch, wild_one) {
  EXPECT_EQ(FNM_NOMATCH, fnmatch("ab?", "ab", 0));
  EXPECT_EQ(0, fnmatch("ab?", "abc", 0));
  EXPECT_EQ(FNM_NOMATCH, fnmatch("ab?", "abcd", 0));
  EXPECT_EQ(0, fnmatch("ab?d", "abcd", 0));
  EXPECT_EQ(0, fnmatch("ab?cd", "ab/cd", 0));
  EXPECT_EQ(FNM_NOMATCH, fnmatch("ab?cd", "ab/cd", FNM_PATHNAME));
}

TEST(fnmatch, leading_dir) {
  EXPECT_EQ(FNM_NOMATCH, fnmatch("ab", "abcd", FNM_LEADING_DIR));
  EXPECT_EQ(0, fnmatch("ab*", "abcd", FNM_LEADING_DIR));
  EXPECT_EQ(0, fnmatch("*ab*", "1/2/3/4/abcd", FNM_LEADING_DIR));
  EXPECT_EQ(FNM_NOMATCH, fnmatch("*ab*", "1/2/3/4/abcd", FNM_PATHNAME | FNM_LEADING_DIR));
  EXPECT_EQ(FNM_NOMATCH, fnmatch("ab?", "abcd", FNM_LEADING_DIR));

  EXPECT_EQ(0, fnmatch("ab", "ab/cd", FNM_LEADING_DIR));
  EXPECT_EQ(0, fnmatch("ab", "ab/cd", FNM_PATHNAME | FNM_LEADING_DIR));
  // TODO(b/175302045) fix this case and enable this test.
  // EXPECT_EQ(0, fnmatch("*ab", "1/2/3/4/ab/cd", FNM_LEADING_DIR));
  EXPECT_EQ(FNM_NOMATCH, fnmatch("*ab", "1/2/3/4/ab/cd", FNM_PATHNAME | FNM_LEADING_DIR));
  EXPECT_EQ(0, fnmatch("ab*", "ab/cd/ef", FNM_LEADING_DIR));
  EXPECT_EQ(0, fnmatch("ab*", "ab/cd/ef", FNM_PATHNAME | FNM_LEADING_DIR));
  EXPECT_EQ(0, fnmatch("*ab*", "1/2/3/4/ab/cd/ef", FNM_LEADING_DIR));
  EXPECT_EQ(FNM_NOMATCH, fnmatch("*ab*", "1/2/3/4/ab/cd/ef", FNM_PATHNAME | FNM_LEADING_DIR));
  EXPECT_EQ(FNM_NOMATCH, fnmatch("ab?", "ab/cd/ef", FNM_LEADING_DIR));
  EXPECT_EQ(0, fnmatch("ab?", "abx/cd/ef", FNM_LEADING_DIR));

  EXPECT_EQ(FNM_NOMATCH, fnmatch("ab/", "ab/cd/ef", FNM_LEADING_DIR));
  EXPECT_EQ(0, fnmatch("ab/*", "ab/cd/ef", FNM_LEADING_DIR));
  EXPECT_EQ(FNM_NOMATCH, fnmatch("ab/?", "ab/cd/ef", FNM_LEADING_DIR));

  // TODO(b/175302045) fix this case and enable this test.
  // EXPECT_EQ(0, fnmatch("ab*c", "ab/1/2/3/c/d/e", FNM_LEADING_DIR));
  EXPECT_EQ(0, fnmatch("ab?c", "ab/c/ef", FNM_LEADING_DIR));

  EXPECT_EQ(0, fnmatch("ab*c*", "ab/1/2/3/c/d/e", FNM_LEADING_DIR));
  EXPECT_EQ(0, fnmatch("ab?c*", "ab/c/ef", FNM_LEADING_DIR));

  EXPECT_EQ(FNM_NOMATCH, fnmatch("ab*c/", "ab/1/2/3/c/d/e", FNM_LEADING_DIR));
  EXPECT_EQ(FNM_NOMATCH, fnmatch("ab?c/", "ab/c/ef", FNM_LEADING_DIR));

  EXPECT_EQ(0, fnmatch("ab*c/*", "ab/1/2/3/c/d/e", FNM_LEADING_DIR));
  EXPECT_EQ(0, fnmatch("ab?c/*", "ab/c/ef", FNM_LEADING_DIR));
}
