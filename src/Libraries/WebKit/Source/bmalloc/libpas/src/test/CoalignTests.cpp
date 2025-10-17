/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 8, 2025.
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
#include "pas_coalign.h"

using namespace std;

namespace {

void testCoalignOneSided(uintptr_t begin, uintptr_t typeSize, uintptr_t alignment, uintptr_t result)
{
    pas_coalign_result actualResult = pas_coalign_one_sided(begin, typeSize, alignment);
    CHECK(actualResult.has_result);
    CHECK_EQUAL(actualResult.result, result);
    CHECK(!(actualResult.result % alignment));
    CHECK(!((actualResult.result - begin) % typeSize));
}

void testCoalignOneSidedError(uintptr_t begin, uintptr_t typeSize, uintptr_t alignment)
{
    pas_coalign_result actualResult = pas_coalign_one_sided(begin, typeSize, alignment);
    CHECK(!actualResult.has_result);
}

void testCoalign(uintptr_t beginLeft, uintptr_t leftSize,
                 uintptr_t beginRight, uintptr_t rightSize,
                 uintptr_t result)
{
    pas_coalign_result actualResult = pas_coalign(beginLeft, leftSize, beginRight, rightSize);
    CHECK(actualResult.has_result);
    CHECK_EQUAL(actualResult.result, result);
    CHECK(!((actualResult.result - beginLeft) % leftSize));
    CHECK(!((actualResult.result - beginRight) % rightSize));
}

} // anonymous namespace

void addCoalignTests()
{
    ADD_TEST(testCoalignOneSided(459318, 666, 1024, 795648));
    ADD_TEST(testCoalignOneSided(4096 + 48 * 4, 48, 64, 4288));
    ADD_TEST(testCoalignOneSided(4096 * 11 + 7 * 13, 13, 1024, 58368));
    ADD_TEST(testCoalignOneSided(4096 * 13 + 7 * 64 * 11, 7 * 64, 1024, 60416));
    ADD_TEST(testCoalignOneSided(10000, 5000, 4096, 2560000));
    ADD_TEST(testCoalignOneSidedError(11111, 5000, 4096));
    ADD_TEST(testCoalignOneSidedError(6422642, 40, 256));
    ADD_TEST(testCoalignOneSided(6422648, 40, 256, 6423808));
    ADD_TEST(testCoalign(500, 155, 999, 1024, 61415));
    ADD_TEST(testCoalign(6426, 531, 24, 647, 219357));
}

