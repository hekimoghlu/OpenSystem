/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 2, 2023.
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

#pragma once

#include <utility>
#include <string>

using PairInts = std::pair<int, int>;
using PairStrings = std::pair<std::string, std::string>;

inline const PairInts &getIntPairPointer() {
    static PairInts value = { 4, 9 };
    return value;
}

inline PairInts getIntPair() {
    return { -5, 12 };
}

struct StructInPair {
    int x;
    int y;
};

using PairStructInt = std::pair<StructInPair, int>;

inline PairStructInt getPairStructInt(int x) {
    return { { x * 2, -x}, x };
}

struct UnsafeStruct {
    int *ptr;
};

struct __attribute__((language_attr("import_iterator"))) Iterator {};

using PairUnsafeStructInt = std::pair<UnsafeStruct, int>;
using PairIteratorInt = std::pair<Iterator, int>;

struct __attribute__((language_attr("import_owned"))) HasMethodThatReturnsUnsafePair {
    PairUnsafeStructInt getUnsafePair() const { return {}; }
    PairIteratorInt getIteratorPair() const { return {}; }
};
