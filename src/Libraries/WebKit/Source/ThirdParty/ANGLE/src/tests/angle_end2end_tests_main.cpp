/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 15, 2024.
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

#include "gtest/gtest.h"
#if defined(ANGLE_HAS_RAPIDJSON)
#    include "test_utils/runner/TestSuite.h"
#endif  // defined(ANGLE_HAS_RAPIDJSON)
#include "util/OSWindow.h"

void ANGLEProcessTestArgs(int *argc, char *argv[]);

// Register* functions handle setting up special tests that need complex parameterization.
// GoogleTest relies heavily on static initialization to register test functions. This can
// cause all sorts of issues if the right variables aren't initialized in the right order.
// This register function needs to be called explicitly after static initialization and
// before the test launcher starts. This is a safer and generally better way to initialize
// tests. It's also more similar to how the dEQP Test harness works. In the future we should
// likely specialize more register functions more like dEQP instead of relying on static init.
void RegisterContextCompatibilityTests();

namespace
{
bool HasArg(int argc, char **argv, const char *arg)
{
    for (int i = 1; i < argc; ++i)
    {
        if (strstr(argv[i], arg) != nullptr)
        {
            return true;
        }
    }
    return false;
}
}  // namespace

int main(int argc, char **argv)
{
    if (!HasArg(argc, argv, "--list-tests") && !HasArg(argc, argv, "--gtest_list_tests") &&
        HasArg(argc, argv, "--use-gl"))
    {
        std::cerr << "--use-gl isn't supported by end2end tests - use *_EGL configs instead "
                     "(angle_test_enable_system_egl=true)\n";
        return EXIT_FAILURE;
    }

    // TODO(b/279980674): TestSuite depends on rapidjson which we don't have in aosp builds,
    // for now disable both TestSuite and expectations.
#if defined(ANGLE_HAS_RAPIDJSON)
    ANGLEProcessTestArgs(&argc, argv);

    auto registerTestsCallback = [] {
        if (!IsTSan())
        {
            RegisterContextCompatibilityTests();
        }
    };
    angle::TestSuite testSuite(&argc, argv, registerTestsCallback);

    constexpr char kTestExpectationsPath[] = "src/tests/angle_end2end_tests_expectations.txt";
    constexpr size_t kMaxPath = 512;
    std::array<char, kMaxPath> foundDataPath;
    if (!angle::FindTestDataPath(kTestExpectationsPath, foundDataPath.data(), foundDataPath.size()))
    {
        std::cerr << "Unable to find test expectations path (" << kTestExpectationsPath << ")\n";
        return EXIT_FAILURE;
    }

    // end2end test expectations only allow SKIP at the moment.
    testSuite.setTestExpectationsAllowMask(angle::GPUTestExpectationsParser::kGpuTestSkip |
                                           angle::GPUTestExpectationsParser::kGpuTestTimeout);

    if (!testSuite.loadAllTestExpectationsFromFile(std::string(foundDataPath.data())))
    {
        return EXIT_FAILURE;
    }

    return testSuite.run();
#else
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
#endif  // defined(ANGLE_HAS_RAPIDJSON)
}
