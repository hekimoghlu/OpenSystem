/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 8, 2023.
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
// Copyright 2017 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// ProgramParameterTest: validate parameters of ProgramParameter

#include "test_utils/ANGLETest.h"

using namespace angle;

namespace
{

class ProgramParameterTest : public ANGLETest<>
{
  protected:
    ProgramParameterTest()
    {
        setWindowWidth(64);
        setWindowHeight(64);
        setConfigRedBits(8);
        setConfigGreenBits(8);
        setConfigBlueBits(8);
        setConfigAlphaBits(8);
    }
};

class ProgramParameterTestES31 : public ProgramParameterTest
{
  protected:
    ProgramParameterTestES31() : ProgramParameterTest() {}
};

// If es version < 3.1, PROGRAM_SEPARABLE is not supported.
TEST_P(ProgramParameterTest, ValidatePname)
{
    GLuint program = glCreateProgram();
    ASSERT_NE(program, 0u);

    glProgramParameteri(program, GL_PROGRAM_BINARY_RETRIEVABLE_HINT, GL_TRUE);
    ASSERT_GL_NO_ERROR();

    glProgramParameteri(program, GL_PROGRAM_SEPARABLE, GL_TRUE);
    if (getClientMajorVersion() < 3 || getClientMinorVersion() < 1)
    {
        ASSERT_GL_ERROR(GL_INVALID_ENUM);
    }
    else
    {
        ASSERT_GL_NO_ERROR();
    }

    glDeleteProgram(program);
}

// Validate parameters for ProgramParameter when pname is PROGRAM_SEPARABLE.
TEST_P(ProgramParameterTestES31, ValidateParameters)
{
    GLuint program = glCreateProgram();
    ASSERT_NE(program, 0u);

    glProgramParameteri(0, GL_PROGRAM_SEPARABLE, GL_TRUE);
    ASSERT_GL_ERROR(GL_INVALID_VALUE);

    glProgramParameteri(program, GL_PROGRAM_SEPARABLE, 2);
    ASSERT_GL_ERROR(GL_INVALID_VALUE);

    glDeleteProgram(program);
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(ProgramParameterTest);
ANGLE_INSTANTIATE_TEST_ES3_AND_ES31(ProgramParameterTest);

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(ProgramParameterTestES31);
ANGLE_INSTANTIATE_TEST_ES31(ProgramParameterTestES31);
}  // namespace
