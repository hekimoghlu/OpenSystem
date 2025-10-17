/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 9, 2023.
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
#ifndef RTC_BASE_GTEST_PROD_UTIL_H_
#define RTC_BASE_GTEST_PROD_UTIL_H_

// Define our own version of FRIEND_TEST here rather than including
// gtest_prod.h to avoid depending on any part of GTest in production code.
#define FRIEND_TEST_WEBRTC(test_case_name, test_name) \
  friend class test_case_name##_##test_name##_Test

// This file is a plain copy of Chromium's base/gtest_prod_util.h.
//
// This is a wrapper for gtest's FRIEND_TEST macro that friends
// test with all possible prefixes. This is very helpful when changing the test
// prefix, because the friend declarations don't need to be updated.
//
// Example usage:
//
// class MyClass {
//  private:
//   void MyMethod();
//   FRIEND_TEST_ALL_PREFIXES(MyClassTest, MyMethod);
// };
#define FRIEND_TEST_ALL_PREFIXES(test_case_name, test_name) \
  FRIEND_TEST_WEBRTC(test_case_name, test_name);            \
  FRIEND_TEST_WEBRTC(test_case_name, DISABLED_##test_name); \
  FRIEND_TEST_WEBRTC(test_case_name, FLAKY_##test_name);    \
  FRIEND_TEST_WEBRTC(test_case_name, FAILS_##test_name)

#endif  // RTC_BASE_GTEST_PROD_UTIL_H_
