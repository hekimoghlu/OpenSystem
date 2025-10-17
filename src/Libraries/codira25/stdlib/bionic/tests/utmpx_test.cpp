/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 3, 2023.
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

#include <utmpx.h>

TEST(utmpx, smoke) {
  // Our utmpx "implementation" just calls the utmp no-op functions.
  setutxent();
  utmpx empty = {.ut_type = EMPTY};
  ASSERT_EQ(NULL, getutxent());
  ASSERT_EQ(NULL, getutxid(&empty));
  ASSERT_EQ(NULL, getutxline(&empty));
  endutxent();
  ASSERT_EQ(NULL, pututxline(&empty));
}
