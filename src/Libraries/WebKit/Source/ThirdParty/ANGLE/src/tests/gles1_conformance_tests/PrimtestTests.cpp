/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 18, 2022.
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
// Copyright 2020 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// PrimtestTests.cpp:
//   GLES1 conformance primtest tests.
//

#include "GLES/gl.h"
#include "test_utils/ANGLETest.h"
#include "test_utils/gl_raii.h"

#ifdef __cplusplus
extern "C" {
#endif

// ES 1.0
extern void DrawPrims(void);

#include "primtest/driver.h"
#include "primtest/tproto.h"

#ifdef __cplusplus
}

#endif

namespace angle
{
class GLES1PrimtestTest : public ANGLETest<>
{
  protected:
    GLES1PrimtestTest()
    {
        setWindowWidth(48);
        setWindowHeight(48);
        setConfigRedBits(8);
        setConfigGreenBits(8);
        setConfigBlueBits(8);
        setConfigAlphaBits(8);
        setConfigDepthBits(24);
        setConfigStencilBits(8);
    }

    void testTearDown() override
    {
        if (mTestData != nullptr)
        {
            free(mTestData);
        }
    }

    void execTest(long test)
    {
        long i;
        for (i = 0; driver[i].test != TEST_NULL; i++)
        {
            if (driver[i].test == test)
            {
                break;
            }
        }

        ASSERT_NE(TEST_NULL, driver[i].test);

        driverRec &op = driver[i];

        op.funcInit((void *)&mTestData);
        op.finish = 0;
        ASSERT_GL_NO_ERROR();

        for (;;)
        {
            op.funcStatus(1, mTestData);
            ASSERT_GL_NO_ERROR();

            op.funcSet(1, mTestData);
            ASSERT_GL_NO_ERROR();

            DrawPrims();
            ASSERT_GL_NO_ERROR();

            long finish = op.funcUpdate(mTestData);
            ASSERT_GL_NO_ERROR();
            if (finish)
            {
                break;
            }
        };
    }

  protected:
    void *mTestData;
};

TEST_P(GLES1PrimtestTest, Hint)
{
    execTest(TEST_HINT);
}

TEST_P(GLES1PrimtestTest, Alias)
{
    execTest(TEST_ALIAS);
}

TEST_P(GLES1PrimtestTest, Alpha)
{
    execTest(TEST_ALPHA);
}

TEST_P(GLES1PrimtestTest, Blend)
{
    execTest(TEST_BLEND);
}

TEST_P(GLES1PrimtestTest, Depth)
{
    execTest(TEST_DEPTH);
}

TEST_P(GLES1PrimtestTest, Dither)
{
    execTest(TEST_DITHER);
}

TEST_P(GLES1PrimtestTest, Fog)
{
    execTest(TEST_FOG);
}

TEST_P(GLES1PrimtestTest, Light)
{
    execTest(TEST_LIGHT);
}

TEST_P(GLES1PrimtestTest, Logic)
{
    execTest(TEST_LOGICOP);
}

TEST_P(GLES1PrimtestTest, Scissor)
{
    execTest(TEST_SCISSOR);
}

TEST_P(GLES1PrimtestTest, Shade)
{
    execTest(TEST_SHADE);
}

TEST_P(GLES1PrimtestTest, Stencil)
{
    execTest(TEST_STENCIL);
}

TEST_P(GLES1PrimtestTest, Texture)
{
    execTest(TEST_TEXTURE);
}

ANGLE_INSTANTIATE_TEST(GLES1PrimtestTest, ES1_OPENGL(), ES1_VULKAN());
}  // namespace angle
