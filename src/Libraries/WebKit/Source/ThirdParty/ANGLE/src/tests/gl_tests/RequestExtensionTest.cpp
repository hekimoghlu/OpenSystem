/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 7, 2024.
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
// Copyright 2018 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// RequestExtensionTest:
//   Tests that extensions can be requested and are disabled by default when using
//   EGL_ANGLE_request_extension
//

#include "test_utils/ANGLETest.h"

#include "util/EGLWindow.h"

namespace angle
{

class RequestExtensionTest : public ANGLETest<>
{
  protected:
    RequestExtensionTest() { setExtensionsEnabled(false); }
};

// Test that a known requestable extension is disabled by default and make sure it can be requested
// if possible
TEST_P(RequestExtensionTest, ExtensionsDisabledByDefault)
{
    EXPECT_TRUE(IsEGLDisplayExtensionEnabled(getEGLWindow()->getDisplay(),
                                             "EGL_ANGLE_create_context_extensions_enabled"));
    EXPECT_FALSE(IsGLExtensionEnabled("GL_OES_rgb8_rgba8"));

    if (IsGLExtensionRequestable("GL_OES_rgb8_rgba8"))
    {
        glRequestExtensionANGLE("GL_OES_rgb8_rgba8");
        EXPECT_TRUE(IsGLExtensionEnabled("GL_OES_rgb8_rgba8"));
    }
}

// Test the queries for the requestable extension strings
TEST_P(RequestExtensionTest, Queries)
{
    ANGLE_SKIP_TEST_IF(!IsGLExtensionEnabled("GL_ANGLE_request_extension"));

    const GLubyte *requestableExtString = glGetString(GL_REQUESTABLE_EXTENSIONS_ANGLE);
    EXPECT_GL_NO_ERROR();
    EXPECT_NE(nullptr, requestableExtString);

    if (getClientMajorVersion() >= 3)
    {
        GLint numExtensions = 0;
        glGetIntegerv(GL_NUM_REQUESTABLE_EXTENSIONS_ANGLE, &numExtensions);
        EXPECT_GL_NO_ERROR();

        for (GLint extIdx = 0; extIdx < numExtensions; extIdx++)
        {
            const GLubyte *requestableExtIndexedString =
                glGetStringi(GL_REQUESTABLE_EXTENSIONS_ANGLE, extIdx);
            EXPECT_GL_NO_ERROR();
            EXPECT_NE(nullptr, requestableExtIndexedString);
        }

        // Request beyond the end of the array
        const GLubyte *requestableExtIndexedString =
            glGetStringi(GL_REQUESTABLE_EXTENSIONS_ANGLE, numExtensions);
        EXPECT_GL_ERROR(GL_INVALID_VALUE);
        EXPECT_EQ(nullptr, requestableExtIndexedString);
    }
}

// Use this to select which configurations (e.g. which renderer, which GLES major version) these
// tests should be run against.
ANGLE_INSTANTIATE_TEST_ES2_AND_ES3(RequestExtensionTest);

}  // namespace angle
