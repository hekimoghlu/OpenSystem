/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 21, 2025.
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
#include "TestHarness.h"
#include "pas_extended_gcd.h"

using namespace std;

namespace {

void testExtendedGCD(int64_t left, int64_t right,
                     int64_t result, int64_t leftBezoutCoefficient, int64_t rightBezoutCoefficient)
{
    pas_extended_gcd_result actualResult = pas_extended_gcd(left, right);
    CHECK_EQUAL(actualResult.result, result);
    CHECK_EQUAL(actualResult.left_bezout_coefficient, leftBezoutCoefficient);
    CHECK_EQUAL(actualResult.right_bezout_coefficient, rightBezoutCoefficient);
    
    actualResult = pas_extended_gcd(right, left);
    CHECK_EQUAL(actualResult.result, result);
    CHECK_EQUAL(actualResult.left_bezout_coefficient, rightBezoutCoefficient);
    CHECK_EQUAL(actualResult.right_bezout_coefficient, leftBezoutCoefficient);
}

} // anonymous namespace

void addExtendedGCDTests()
{
    ADD_TEST(testExtendedGCD(1, 0, 1, 1, 0));
    ADD_TEST(testExtendedGCD(2, 0, 2, 1, 0));
    ADD_TEST(testExtendedGCD(1, 64, 1, 1, 0));
    ADD_TEST(testExtendedGCD(666, 42, 6, -1, 16));
    ADD_TEST(testExtendedGCD(666, 1024, 2, -123, 80));
    ADD_TEST(testExtendedGCD(65536, 32768, 32768, 0, 1));
}

