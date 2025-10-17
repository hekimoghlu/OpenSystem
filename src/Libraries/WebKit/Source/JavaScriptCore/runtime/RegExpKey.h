/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 10, 2025.
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

#include "YarrFlags.h"
#include <wtf/OptionSet.h>
#include <wtf/PackedRefPtr.h>
#include <wtf/text/StringHash.h>

namespace JSC {

struct RegExpKey {
    OptionSet<Yarr::Flags> flagsValue;
    PackedRefPtr<StringImpl> pattern;

    RegExpKey()
    {
    }

    RegExpKey(OptionSet<Yarr::Flags> flags)
        : flagsValue(flags)
    {
    }

    RegExpKey(OptionSet<Yarr::Flags> flags, const String& pattern)
        : flagsValue(flags)
        , pattern(pattern.impl())
    {
    }

    RegExpKey(OptionSet<Yarr::Flags> flags, RefPtr<StringImpl>&& pattern)
        : flagsValue(flags)
        , pattern(WTFMove(pattern))
    {
    }

    RegExpKey(OptionSet<Yarr::Flags> flags, const RefPtr<StringImpl>& pattern)
        : flagsValue(flags)
        , pattern(pattern)
    {
    }

    friend inline bool operator==(const RegExpKey& a, const RegExpKey& b);

    struct Hash {
        static unsigned hash(const RegExpKey& key) { return key.pattern->hash(); }
        static bool equal(const RegExpKey& a, const RegExpKey& b) { return a == b; }
        static constexpr bool safeToCompareToEmptyOrDeleted = false;
        static constexpr bool hasHashInValue = true;
    };
};

inline bool operator==(const RegExpKey& a, const RegExpKey& b)
{
    if (a.flagsValue != b.flagsValue)
        return false;
    if (!a.pattern)
        return !b.pattern;
    if (!b.pattern)
        return false;
    return equal(a.pattern.get(), b.pattern.get());
}

} // namespace JSC

namespace WTF {
template<typename> struct DefaultHash;

template<> struct DefaultHash<JSC::RegExpKey> : JSC::RegExpKey::Hash { };

template<> struct HashTraits<JSC::RegExpKey> : GenericHashTraits<JSC::RegExpKey> {
    static constexpr bool emptyValueIsZero = true;
    static void constructDeletedValue(JSC::RegExpKey& slot) { slot.flagsValue = JSC::Yarr::Flags::DeletedValue; }
    static bool isDeletedValue(const JSC::RegExpKey& value) { return value.flagsValue == JSC::Yarr::Flags::DeletedValue; }
};
} // namespace WTF
