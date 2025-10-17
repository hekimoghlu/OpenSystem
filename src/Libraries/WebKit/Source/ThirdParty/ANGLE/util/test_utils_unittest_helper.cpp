/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 20, 2024.
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

// system_utils_unittest_helper.cpp: Helper to the SystemUtils.RunApp unittest

#include "test_utils_unittest_helper.h"

#include "../src/tests/test_utils/runner/TestSuite.h"
#include "common/system_utils.h"

#include <string.h>

// This variable is also defined in angle_unittest_main.
bool gVerbose = false;

int main(int argc, char **argv)
{
    bool runTestSuite = false;

    for (int argIndex = 1; argIndex < argc; ++argIndex)
    {
        if (strcmp(argv[argIndex], kRunTestSuite) == 0)
        {
            runTestSuite = true;
        }
    }

    if (runTestSuite)
    {
        angle::TestSuite testSuite(&argc, argv);
        return testSuite.run();
    }

    if (argc != 3 || strcmp(argv[1], kRunAppTestArg1) != 0 || strcmp(argv[2], kRunAppTestArg2) != 0)
    {
        fprintf(stderr, "Expected command line:\n%s %s %s\n", argv[0], kRunAppTestArg1,
                kRunAppTestArg2);
        return EXIT_FAILURE;
    }

    std::string env = angle::GetEnvironmentVar(kRunAppTestEnvVarName);
    if (env == "")
    {
        printf("%s", kRunAppTestStdout);
        fflush(stdout);
        fprintf(stderr, "%s", kRunAppTestStderr);
    }
    else
    {
        fprintf(stderr, "%s", env.c_str());
    }

    return EXIT_SUCCESS;
}
