/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 8, 2024.
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
// Copyright 2024 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// EGLNoErrorTest.cpp:
//   Tests for the EGL extension EGL_ANGLE_no_error
//

#include <gtest/gtest.h>

#include "test_utils/ANGLETest.h"

using namespace angle;

class EGLNoErrorTest : public ANGLETest<>
{};

// Validation errors become undefined behavour with this extension. Simply test turning validation
// off and on.
TEST_P(EGLNoErrorTest, EnableDisable)
{
    if (IsEGLClientExtensionEnabled("EGL_ANGLE_no_error"))
    {
        eglSetValidationEnabledANGLE(EGL_FALSE);
        eglSetValidationEnabledANGLE(EGL_TRUE);
        EXPECT_EGL_ERROR(EGL_SUCCESS);
    }
    else
    {
        eglSetValidationEnabledANGLE(EGL_FALSE);
        EXPECT_EGL_ERROR(EGL_BAD_ACCESS);
    }
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(EGLNoErrorTest);
ANGLE_INSTANTIATE_TEST(EGLNoErrorTest,
                       ES2_D3D9(),
                       ES2_D3D11(),
                       ES3_D3D11(),
                       ES2_OPENGL(),
                       ES3_OPENGL(),
                       ES2_VULKAN());
