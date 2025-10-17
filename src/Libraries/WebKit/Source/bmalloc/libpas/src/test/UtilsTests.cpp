/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 28, 2025.
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
#include "pas_utils.h"

using namespace std;

namespace {

void testIsDivisibleBy3(unsigned value)
{
    static const uint64_t magic_constant = PAS_IS_DIVISIBLE_BY_MAGIC_CONSTANT(3);
    CHECK_EQUAL(pas_is_divisible_by(value, magic_constant),
                !(value % 3));
}

} // anonymous namespace

void addUtilsTests()
{
    ADD_TEST(testIsDivisibleBy3(0));
    ADD_TEST(testIsDivisibleBy3(1));
    ADD_TEST(testIsDivisibleBy3(2));
    ADD_TEST(testIsDivisibleBy3(3));
    ADD_TEST(testIsDivisibleBy3(4));
    ADD_TEST(testIsDivisibleBy3(5));
    ADD_TEST(testIsDivisibleBy3(6));
    ADD_TEST(testIsDivisibleBy3(7));
    ADD_TEST(testIsDivisibleBy3(8));
    ADD_TEST(testIsDivisibleBy3(9));
    ADD_TEST(testIsDivisibleBy3(10));
    ADD_TEST(testIsDivisibleBy3(11));
    ADD_TEST(testIsDivisibleBy3(12));
    ADD_TEST(testIsDivisibleBy3(13));
    ADD_TEST(testIsDivisibleBy3(14));
    ADD_TEST(testIsDivisibleBy3(15));
    ADD_TEST(testIsDivisibleBy3(16));
    ADD_TEST(testIsDivisibleBy3(17));
    ADD_TEST(testIsDivisibleBy3(18));
    ADD_TEST(testIsDivisibleBy3(19));
    ADD_TEST(testIsDivisibleBy3(20));
    ADD_TEST(testIsDivisibleBy3(21));
    ADD_TEST(testIsDivisibleBy3(22));
    ADD_TEST(testIsDivisibleBy3(23));
    ADD_TEST(testIsDivisibleBy3(24));
    ADD_TEST(testIsDivisibleBy3(25));
    ADD_TEST(testIsDivisibleBy3(26));
    ADD_TEST(testIsDivisibleBy3(27));
}
