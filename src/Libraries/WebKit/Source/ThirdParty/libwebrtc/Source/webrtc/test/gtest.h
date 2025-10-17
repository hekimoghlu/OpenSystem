/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 16, 2023.
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
#ifndef TEST_GTEST_H_
#define TEST_GTEST_H_

#include "rtc_base/ignore_wundef.h"

RTC_PUSH_IGNORING_WUNDEF()
#ifdef WEBRTC_WEBKIT_BUILD
#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>
#else
#include "testing/gtest/include/gtest/gtest-spi.h"
#include "testing/gtest/include/gtest/gtest.h"
#endif // WEBRTC_WEBKIT_BUILD
RTC_POP_IGNORING_WUNDEF()

// GTEST_HAS_DEATH_TEST is set to 1 when death tests are supported, but appears
// to be left unset if they're not supported. Rather than depend on this, we
// set it to 0 ourselves here.
#ifndef GTEST_HAS_DEATH_TEST
#define GTEST_HAS_DEATH_TEST 0
#endif

#endif  // TEST_GTEST_H_
