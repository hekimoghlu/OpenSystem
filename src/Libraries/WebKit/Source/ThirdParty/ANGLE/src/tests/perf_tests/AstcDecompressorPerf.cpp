/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 18, 2025.
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
// AstcDecompressorPerf: Performance test for the ASTC decompressor.
//

#include "ANGLEPerfTest.h"

#include <gmock/gmock.h>

#include "common/WorkerThread.h"
#include "image_util/AstcDecompressor.h"
#include "image_util/AstcDecompressorTestUtils.h"

using namespace testing;

namespace
{
using angle::AstcDecompressor;
using angle::WorkerThreadPool;

struct AstcDecompressorParams
{
    AstcDecompressorParams(uint32_t width, uint32_t height) : width(width), height(height) {}

    uint32_t width;
    uint32_t height;
};

std::ostream &operator<<(std::ostream &os, const AstcDecompressorParams &params)
{
    os << params.width << "x" << params.height;
    return os;
}

class AstcDecompressorPerfTest : public ANGLEPerfTest,
                                 public WithParamInterface<AstcDecompressorParams>
{
  public:
    AstcDecompressorPerfTest();

    void step() override;

    std::string getName();

    AstcDecompressor &mDecompressor;
    std::vector<uint8_t> mInput;
    std::vector<uint8_t> mOutput;
    std::shared_ptr<WorkerThreadPool> mSingleThreadPool;
    std::shared_ptr<WorkerThreadPool> mMultiThreadPool;
};

AstcDecompressorPerfTest::AstcDecompressorPerfTest()
    : ANGLEPerfTest(getName(), "", "_run", 1, "us"),
      mDecompressor(AstcDecompressor::get()),
      mInput(makeAstcCheckerboard(GetParam().width, GetParam().height)),
      mOutput(GetParam().width * GetParam().height * 4),
      mSingleThreadPool(WorkerThreadPool::Create(1, ANGLEPlatformCurrent())),
      mMultiThreadPool(WorkerThreadPool::Create(0, ANGLEPlatformCurrent()))
{}

void AstcDecompressorPerfTest::step()
{
    mDecompressor.decompress(mSingleThreadPool, mMultiThreadPool, GetParam().width,
                             GetParam().height, 8, 8, mInput.data(), mInput.size(), mOutput.data());
}

std::string AstcDecompressorPerfTest::getName()
{
    std::stringstream ss;
    ss << UnitTest::GetInstance()->current_test_suite()->name() << "/" << GetParam();
    return ss.str();
}

// Measures the speed of ASTC decompression on the CPU.
TEST_P(AstcDecompressorPerfTest, Run)
{
    if (!mDecompressor.available())
        skipTest("ASTC decompressor not available");

    this->run();
}

INSTANTIATE_TEST_SUITE_P(,
                         AstcDecompressorPerfTest,
                         Values(AstcDecompressorParams(16, 16),
                                AstcDecompressorParams(256, 256),
                                AstcDecompressorParams(1024, 1024)),
                         PrintToStringParamName());

}  // anonymous namespace
