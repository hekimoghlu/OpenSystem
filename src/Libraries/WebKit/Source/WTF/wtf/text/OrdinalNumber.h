/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 16, 2022.
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

#include <compare>
#include <wtf/HashFunctions.h>
#include <wtf/HashTraits.h>

namespace WTF {

// An abstract number of element in a sequence. The sequence has a first element.
// This type should be used instead of integer because 2 contradicting traditions can
// call a first element '0' or '1' which makes integer type ambiguous.
class OrdinalNumber {
    WTF_MAKE_FAST_ALLOCATED;
public:
    static OrdinalNumber beforeFirst() { return OrdinalNumber(-1); }
    static OrdinalNumber fromZeroBasedInt(int zeroBasedInt) { return OrdinalNumber(zeroBasedInt); }
    static OrdinalNumber fromOneBasedInt(int oneBasedInt) { return OrdinalNumber(oneBasedInt - 1); }

    OrdinalNumber() : m_zeroBasedValue(0) { }

    int zeroBasedInt() const { return m_zeroBasedValue; }
    int oneBasedInt() const { return m_zeroBasedValue + 1; }

    friend bool operator==(OrdinalNumber, OrdinalNumber) = default;
    friend std::strong_ordering operator<=>(OrdinalNumber, OrdinalNumber) = default;

private:
    OrdinalNumber(int zeroBasedInt) : m_zeroBasedValue(zeroBasedInt) { }
    int m_zeroBasedValue;
};

template<typename T> struct DefaultHash;
template<> struct DefaultHash<OrdinalNumber> {
    static unsigned hash(OrdinalNumber key) { return intHash(static_cast<unsigned>(key.zeroBasedInt())); }
    static bool equal(OrdinalNumber a, OrdinalNumber b) { return a == b; }
    static constexpr bool safeToCompareToEmptyOrDeleted = true;
};

template<typename T> struct HashTraits;
template<> struct HashTraits<OrdinalNumber> : GenericHashTraits<OrdinalNumber> {
    static void constructDeletedValue(OrdinalNumber& slot)
    {
        slot = OrdinalNumber::beforeFirst();
    }
    static bool isDeletedValue(OrdinalNumber value)
    {
        return value == OrdinalNumber::beforeFirst();
    }
};

}

using WTF::OrdinalNumber;
