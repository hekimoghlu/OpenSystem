/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 14, 2025.
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

//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// system_utils_unittest_helper.h: Constants used by the SystemUtils.RunApp unittest

#ifndef COMMON_SYSTEM_UTILS_UNITTEST_HELPER_H_
#define COMMON_SYSTEM_UTILS_UNITTEST_HELPER_H_

namespace
{
constexpr char kRunAppTestEnvVarName[]  = "RUN_APP_TEST_ENV";
constexpr char kRunAppTestEnvVarValue[] = "RunAppTest environment variable value\n";
constexpr char kRunAppTestStdout[]      = "RunAppTest stdout test\n";
constexpr char kRunAppTestStderr[] = "RunAppTest stderr test\n  .. that expands multiple lines\n";
constexpr char kRunAppTestArg1[]   = "--expected-arg1";
constexpr char kRunAppTestArg2[]   = "expected_arg2";
constexpr char kRunTestSuite[]     = "--run-test-suite";
}  // anonymous namespace

#endif  // COMMON_SYSTEM_UTILS_UNITTEST_HELPER_H_
