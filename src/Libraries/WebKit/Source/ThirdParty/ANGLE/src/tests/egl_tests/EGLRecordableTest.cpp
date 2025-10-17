/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 21, 2025.
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
//
// EGLRecordableTest.cpp:
//   Tests of EGL_ANDROID_recordable extension

#include <gtest/gtest.h>

#include "test_utils/ANGLETest.h"
#include "test_utils/angle_test_configs.h"
#include "util/EGLWindow.h"

using namespace angle;

namespace angle
{
class EGLRecordableTest : public ANGLETest<>
{
  protected:
    EGLRecordableTest() {}
};

// Test that the extension is always available (it is implemented in ANGLE's frontend).
TEST_P(EGLRecordableTest, ExtensionAlwaysAvailable)
{
    EGLDisplay display = getEGLWindow()->getDisplay();
    ASSERT_TRUE(IsEGLDisplayExtensionEnabled(display, "EGL_ANDROID_recordable"));
}

// Check that the default message filters and callbacks are correct
TEST_P(EGLRecordableTest, CheckAllContexts)
{
    EGLDisplay display = getEGLWindow()->getDisplay();
    ANGLE_SKIP_TEST_IF(!IsEGLDisplayExtensionEnabled(display, "EGL_ANDROID_recordable"));

    int nConfigs = 0;
    ASSERT_EGL_TRUE(eglGetConfigs(display, nullptr, 0, &nConfigs));
    ASSERT_NE(nConfigs, 0);

    int nReturnedConfigs = 0;
    std::vector<EGLConfig> configs(nConfigs);
    ASSERT_EGL_TRUE(eglGetConfigs(display, configs.data(), nConfigs, &nReturnedConfigs));
    ASSERT_EQ(nConfigs, nReturnedConfigs);

    for (EGLConfig config : configs)
    {
        EGLint isRecordable;
        eglGetConfigAttrib(display, config, EGL_RECORDABLE_ANDROID, &isRecordable);
    }

    const EGLint configAttributes[] = {
        EGL_RED_SIZE,     EGL_DONT_CARE,  EGL_GREEN_SIZE,         EGL_DONT_CARE,  EGL_BLUE_SIZE,
        EGL_DONT_CARE,    EGL_ALPHA_SIZE, EGL_DONT_CARE,          EGL_DEPTH_SIZE, EGL_DONT_CARE,
        EGL_STENCIL_SIZE, EGL_DONT_CARE,  EGL_RECORDABLE_ANDROID, EGL_TRUE,       EGL_NONE};
    EGLint configCount;
    ASSERT_EGL_TRUE(
        eglChooseConfig(display, configAttributes, configs.data(), configs.size(), &configCount));
    ASSERT_EGL_SUCCESS();
}

}  // namespace angle

ANGLE_INSTANTIATE_TEST_ES2(EGLRecordableTest);
