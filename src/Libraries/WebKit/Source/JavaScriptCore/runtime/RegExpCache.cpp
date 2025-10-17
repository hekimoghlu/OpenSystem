/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 22, 2023.
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
#include "config.h"
#include "RegExpCache.h"

#include "StrongInlines.h"
#include <wtf/TZoneMallocInlines.h>

namespace JSC {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RegExpCache);

RegExp* RegExpCache::lookupOrCreate(VM& vm, const String& patternString, OptionSet<Yarr::Flags> flags)
{
    RegExpKey key(flags, patternString);
    if (RegExp* regExp = m_weakCache.get(key))
        return regExp;

    RegExp* regExp = RegExp::createWithoutCaching(vm, patternString, flags);
#if ENABLE(REGEXP_TRACING)
    vm.addRegExpToTrace(regExp);
#endif

    weakAdd(m_weakCache, key, Weak<RegExp>(regExp, this));
    return regExp;
}

RegExp* RegExpCache::ensureEmptyRegExpSlow(VM& vm)
{
    RegExp* regExp = RegExp::create(vm, emptyString(), { });
    m_emptyRegExp = regExp;
    return regExp;
}

void RegExpCache::finalize(Handle<Unknown> handle, void*)
{
    RegExp* regExp = static_cast<RegExp*>(handle.get().asCell());
    weakRemove(m_weakCache, regExp->key(), regExp);
}

void RegExpCache::addToStrongCache(RegExp* regExp)
{
    String pattern = regExp->pattern();
    if (pattern.length() > maxStrongCacheablePatternLength)
        return;

    m_strongCache[m_nextEntryInStrongCache] = regExp;
    m_nextEntryInStrongCache++;
    if (m_nextEntryInStrongCache == maxStrongCacheableEntries)
        m_nextEntryInStrongCache = 0;
}

void RegExpCache::deleteAllCode()
{
    m_strongCache.fill(nullptr);
    m_nextEntryInStrongCache = 0;

    for (auto& [key, weakHandle] : m_weakCache) {
        RegExp* regExp = weakHandle.get();
        if (!regExp) // Skip zombies.
            continue;
        regExp->deleteCode();
    }
}

template<typename Visitor>
void RegExpCache::visitAggregateImpl(Visitor& visitor)
{
    for (auto cell : m_strongCache)
        visitor.appendUnbarriered(cell);
    visitor.appendUnbarriered(m_emptyRegExp);
}
DEFINE_VISIT_AGGREGATE(RegExpCache);

}
