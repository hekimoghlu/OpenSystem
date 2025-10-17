/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 17, 2023.
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
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// IndexDataManagerPerfTest:
//   Performance test for index buffer management.
//

#include "ANGLEPerfTest.h"

#include <gmock/gmock.h>

#include "common/bitset_utils.h"

using namespace testing;

namespace
{
template <typename T>
class BitSetIteratorPerfTest : public ANGLEPerfTest
{
  public:
    BitSetIteratorPerfTest();

    void step() override;

    T mBits;
};

template <typename T>
BitSetIteratorPerfTest<T>::BitSetIteratorPerfTest()
    : ANGLEPerfTest(::testing::UnitTest::GetInstance()->current_test_suite()->name(), "", "_run", 1)
{}

template <typename T>
void BitSetIteratorPerfTest<T>::step()
{
    mBits.flip();

    for (size_t bit : mBits)
    {
        ANGLE_UNUSED_VARIABLE(bit);
    }

    mBits.reset();
}

using TestTypes = Types<angle::BitSet32<32>,
                        angle::BitSet64<32>,
                        angle::BitSet64<64>,
                        angle::BitSet<96>,
                        angle::BitSetArray<96>,
                        angle::BitSetArray<300>>;

constexpr char kTestTypeNames[][100] = {"BitSet32_32", "BitSet64_32",    "BitSet64_64",
                                        "BitSet_96",   "BitSetArray_96", "BitSetArray_300"};

class BitSetIteratorTypeNames
{
  public:
    template <typename BitSetType>
    static std::string GetName(int typeIndex)
    {
        return kTestTypeNames[typeIndex];
    }
};

TYPED_TEST_SUITE(BitSetIteratorPerfTest, TestTypes, BitSetIteratorTypeNames);

TYPED_TEST(BitSetIteratorPerfTest, Run)
{
    this->run();
}

}  // anonymous namespace
