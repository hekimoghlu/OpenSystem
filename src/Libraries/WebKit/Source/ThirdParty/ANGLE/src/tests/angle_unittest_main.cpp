/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 9, 2022.
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
// Copyright 2013 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#include "GLSLANG/ShaderLang.h"
#include "gtest/gtest.h"
#include "test_utils/runner/TestSuite.h"

class CompilerTestEnvironment : public testing::Environment
{
  public:
    void SetUp() override
    {
        if (!sh::Initialize())
        {
            FAIL() << "Failed to initialize the compiler.";
        }
    }

    void TearDown() override
    {
        if (!sh::Finalize())
        {
            FAIL() << "Failed to finalize the compiler.";
        }
    }
};

// This variable is also defined in test_utils_unittest_helper.
bool gVerbose = false;

int main(int argc, char **argv)
{
    for (int argIndex = 1; argIndex < argc; ++argIndex)
    {
        if (strcmp(argv[argIndex], "-v") == 0 || strcmp(argv[argIndex], "--verbose") == 0)
        {
            gVerbose = true;
        }
    }

    angle::TestSuite testSuite(&argc, argv);
    testing::AddGlobalTestEnvironment(new CompilerTestEnvironment());
    return testSuite.run();
}
