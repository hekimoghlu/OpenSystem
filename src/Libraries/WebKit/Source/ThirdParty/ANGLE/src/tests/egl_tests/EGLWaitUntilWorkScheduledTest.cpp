/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 8, 2022.
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
// Copyright 2022 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// EGLWaitUntilWorkScheduledTest.cpp:
//   Checks the implementation of EGL_ANGLE_wait_until_work_scheduled.
//

#include <gtest/gtest.h>
#include <tuple>

#include "common/debug.h"
#include "common/string_utils.h"
#include "gpu_info_util/SystemInfo.h"
#include "test_utils/ANGLETest.h"
#include "test_utils/angle_test_platform.h"
#include "test_utils/system_info_util.h"
#include "util/OSWindow.h"

using namespace angle;

class EGLWaitUntilWorkScheduledTest : public ANGLETest<>
{
  public:
    void testSetUp() override { (void)GetSystemInfo(&mSystemInfo); }

  protected:
    EGLDisplay getDisplay() const { return getEGLWindow()->getDisplay(); }

    SystemInfo mSystemInfo;
};

// Test if EGL_ANGLE_wait_until_work_scheduled is enabled that we can call
// eglWaitUntilWorkScheduledANGLE.
TEST_P(EGLWaitUntilWorkScheduledTest, WaitUntilWorkScheduled)
{
    ANGLE_SKIP_TEST_IF(
        !IsEGLDisplayExtensionEnabled(getDisplay(), "EGL_ANGLE_wait_until_work_scheduled"));

    // We're not checking anything except that the function can be called.
    eglWaitUntilWorkScheduledANGLE(getDisplay());
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(EGLWaitUntilWorkScheduledTest);
ANGLE_INSTANTIATE_TEST(EGLWaitUntilWorkScheduledTest,
                       ES2_METAL(),
                       ES3_METAL(),
                       ES2_OPENGL(),
                       ES3_OPENGL());
