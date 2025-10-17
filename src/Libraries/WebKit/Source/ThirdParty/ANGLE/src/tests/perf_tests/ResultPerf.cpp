/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 23, 2024.
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
// ResultPerf:
//   Performance test for ANGLE's Error result class.
//

#include "ANGLEPerfTest.h"
#include "libANGLE/Error.h"

volatile int gThing = 0;

namespace
{
constexpr int kIterationsPerStep = 1000;

class ResultPerfTest : public ANGLEPerfTest
{
  public:
    ResultPerfTest();
    void step() override;
};

ResultPerfTest::ResultPerfTest() : ANGLEPerfTest("ResultPerf", "", "_run", kIterationsPerStep) {}

ANGLE_NOINLINE angle::Result ExternalCall()
{
    if (gThing != 0)
    {
        printf("Something very slow");
        return angle::Result::Stop;
    }
    else
    {
        return angle::Result::Continue;
    }
}

angle::Result CallReturningResult(int depth)
{
    ANGLE_TRY(ExternalCall());
    ANGLE_TRY(ExternalCall());
    ANGLE_TRY(ExternalCall());
    ANGLE_TRY(ExternalCall());
    ANGLE_TRY(ExternalCall());
    ANGLE_TRY(ExternalCall());
    ANGLE_TRY(ExternalCall());
    ANGLE_TRY(ExternalCall());
    ANGLE_TRY(ExternalCall());
    return ExternalCall();
}

void ResultPerfTest::step()
{
    for (int i = 0; i < kIterationsPerStep; i++)
    {
        (void)CallReturningResult(0);
        (void)CallReturningResult(0);
        (void)CallReturningResult(0);
        (void)CallReturningResult(0);
        (void)CallReturningResult(0);
    }
}

TEST_F(ResultPerfTest, Run)
{
    run();
}
}  // anonymous namespace
