/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 1, 2024.
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
#ifndef TEST_TESTSUPPORT_RTC_EXPECT_DEATH_H_
#define TEST_TESTSUPPORT_RTC_EXPECT_DEATH_H_

#include "test/gtest.h"

#if RTC_CHECK_MSG_ENABLED
#define RTC_EXPECT_DEATH(statement, regex) EXPECT_DEATH(statement, regex)
#else
// If RTC_CHECKs messages are disabled we can't validate failure message
#define RTC_EXPECT_DEATH(statement, regex) EXPECT_DEATH(statement, "")
#endif

#endif  // TEST_TESTSUPPORT_RTC_EXPECT_DEATH_H_
