/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 31, 2023.
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
// ConstantFoldingOverflow_test.cpp:
//   Tests for constant folding that results in floating point overflow.
//   In IEEE floating point, the overflow result depends on which of the various rounding modes is
//   chosen - it's either the maximum representable value or infinity.
//   ESSL 3.00.6 section 4.5.1 says that the rounding mode cannot be set and is undefined, so the
//   result in this case is not defined by the spec.
//   We decide to overflow to infinity and issue a warning.
//

#include "tests/test_utils/ConstantFoldingTest.h"

using namespace sh;

namespace
{

class ConstantFoldingOverflowExpressionTest : public ConstantFoldingExpressionTest
{
  public:
    ConstantFoldingOverflowExpressionTest() {}

    void evaluateFloatOverflow(const std::string &floatString, bool positive)
    {
        evaluateFloat(floatString);
        float expected = positive ? std::numeric_limits<float>::infinity()
                                  : -std::numeric_limits<float>::infinity();
        ASSERT_TRUE(constantFoundInAST(expected));
        ASSERT_TRUE(hasWarning());
    }
};

}  // anonymous namespace

// Test that addition that overflows is evaluated correctly.
TEST_F(ConstantFoldingOverflowExpressionTest, Add)
{
    const std::string &floatString = "2.0e38 + 2.0e38";
    evaluateFloatOverflow(floatString, true);
}

// Test that subtraction that overflows is evaluated correctly.
TEST_F(ConstantFoldingOverflowExpressionTest, Subtract)
{
    const std::string &floatString = "2.0e38 - (-2.0e38)";
    evaluateFloatOverflow(floatString, true);
}

// Test that multiplication that overflows is evaluated correctly.
TEST_F(ConstantFoldingOverflowExpressionTest, Multiply)
{
    const std::string &floatString = "1.0e30 * 1.0e10";
    evaluateFloatOverflow(floatString, true);
}

// Test that division that overflows is evaluated correctly.
TEST_F(ConstantFoldingOverflowExpressionTest, Divide)
{
    const std::string &floatString = "1.0e30 / 1.0e-10";
    evaluateFloatOverflow(floatString, true);
}
