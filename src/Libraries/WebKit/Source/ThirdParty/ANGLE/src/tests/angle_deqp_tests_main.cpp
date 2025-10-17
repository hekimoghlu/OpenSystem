/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 10, 2022.
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
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// angle_deqp_gtest_main:
//   Entry point for standalone dEQP tests.

#include <gtest/gtest.h>

#include "test_utils/runner/TestSuite.h"

// Defined in angle_deqp_gtest.cpp. Declared here so we don't need to make a header that we import
// in Chromium.
namespace angle
{
int RunGLCTSTests(int *argc, char **argv);
}  // namespace angle

int main(int argc, char **argv)
{
#if defined(ANGLE_PLATFORM_MACOS)
    // By default, we should hook file API functions on macOS to avoid slow Metal shader caching
    // file access.
    angle::InitMetalFileAPIHooking(argc, argv);
#endif

    return angle::RunGLCTSTests(&argc, argv);
}
