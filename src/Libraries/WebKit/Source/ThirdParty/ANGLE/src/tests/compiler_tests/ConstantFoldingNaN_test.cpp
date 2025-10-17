/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 14, 2022.
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
// Copyright 2016 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// ConstantFoldingNaN_test.cpp:
//   Tests for constant folding that results in NaN according to IEEE and should also generate a
//   warning. The ESSL spec does not mandate generating NaNs, but this is reasonable behavior in
//   this case.
//

#include "tests/test_utils/ConstantFoldingTest.h"

using namespace sh;

namespace
{

class ConstantFoldingNaNExpressionTest : public ConstantFoldingExpressionTest
{
  public:
    ConstantFoldingNaNExpressionTest() {}

    void evaluateFloatNaN(const std::string &floatString)
    {
        evaluateFloat(floatString);
        ASSERT_TRUE(constantFoundInAST(std::numeric_limits<float>::quiet_NaN()));
        ASSERT_TRUE(hasWarning());
    }
};

}  // anonymous namespace

// Test that infinity - infinity evaluates to NaN.
TEST_F(ConstantFoldingNaNExpressionTest, FoldInfinityMinusInfinity)
{
    const std::string &floatString = "1.0e2048 - 1.0e2048";
    evaluateFloatNaN(floatString);
}

// Test that infinity + negative infinity evaluates to NaN.
TEST_F(ConstantFoldingNaNExpressionTest, FoldInfinityPlusNegativeInfinity)
{
    const std::string &floatString = "1.0e2048 + (-1.0e2048)";
    evaluateFloatNaN(floatString);
}

// Test that infinity multiplied by zero evaluates to NaN.
TEST_F(ConstantFoldingNaNExpressionTest, FoldInfinityMultipliedByZero)
{
    const std::string &floatString = "1.0e2048 * 0.0";
    evaluateFloatNaN(floatString);
}

// Test that infinity divided by infinity evaluates to NaN.
TEST_F(ConstantFoldingNaNExpressionTest, FoldInfinityDividedByInfinity)
{
    const std::string &floatString = "1.0e2048 / 1.0e2048";
    evaluateFloatNaN(floatString);
}

// Test that zero divided by zero evaluates to NaN.
TEST_F(ConstantFoldingNaNExpressionTest, FoldZeroDividedByZero)
{
    const std::string &floatString = "0.0 / 0.0";
    evaluateFloatNaN(floatString);
}
