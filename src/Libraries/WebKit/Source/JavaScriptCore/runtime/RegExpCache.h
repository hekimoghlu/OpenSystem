/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 28, 2024.
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

#include "RegExp.h"
#include "RegExpKey.h"
#include "Strong.h"
#include "Weak.h"
#include <array>
#include <wtf/RobinHoodHashMap.h>
#include <wtf/TZoneMalloc.h>

namespace JSC {

namespace Yarr {
enum class Flags : uint16_t;
}

class RegExpCache final : private WeakHandleOwner {
    WTF_MAKE_TZONE_ALLOCATED(RegExpCache);

    friend class RegExp;
    typedef MemoryCompactRobinHoodHashMap<RegExpKey, Weak<RegExp>> RegExpCacheMap;

public:
    RegExpCache() = default;
    void deleteAllCode();

    RegExp* ensureEmptyRegExp(VM& vm)
    {
        if (LIKELY(m_emptyRegExp))
            return m_emptyRegExp;
        return ensureEmptyRegExpSlow(vm);
    }

    DECLARE_VISIT_AGGREGATE;

private:
    static constexpr unsigned maxStrongCacheablePatternLength = 256;

    static constexpr int maxStrongCacheableEntries = 64;

    void finalize(Handle<Unknown>, void* context) final;

    RegExp* ensureEmptyRegExpSlow(VM&);

    RegExp* lookupOrCreate(VM&, const WTF::String& patternString, OptionSet<Yarr::Flags>);
    void addToStrongCache(RegExp*);

    RegExpCacheMap m_weakCache; // Holds all regular expressions currently live.
    unsigned m_nextEntryInStrongCache { 0 };
    std::array<RegExp*, maxStrongCacheableEntries> m_strongCache { }; // Holds a select few regular expressions that have compiled and executed
    RegExp* m_emptyRegExp { nullptr };
};

} // namespace JSC
