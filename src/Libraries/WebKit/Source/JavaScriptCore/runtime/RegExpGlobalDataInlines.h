/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 8, 2023.
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

#include "RegExpGlobalData.h"

namespace JSC {

ALWAYS_INLINE void RegExpCachedResult::record(VM& vm, JSObject* owner, RegExp* regExp, JSString* input, MatchResult result, bool oneCharacterMatch)
{
    m_lastRegExp.setWithoutWriteBarrier(regExp);
    m_lastInput.setWithoutWriteBarrier(input);
    m_result = result;
    m_reified = false;
    m_oneCharacterMatch = oneCharacterMatch;
    vm.writeBarrier(owner);
}

inline void RegExpGlobalData::setInput(JSGlobalObject* globalObject, JSString* string)
{
    m_cachedResult.setInput(globalObject, globalObject, string);
}

/*
   To facilitate result caching, exec(), test(), match(), search(), and replace() dipatch regular
   expression matching through the performMatch function. We use cached results to calculate,
   e.g., RegExp.lastMatch and RegExp.leftParen.
*/
ALWAYS_INLINE MatchResult RegExpGlobalData::performMatch(JSGlobalObject* owner, RegExp* regExp, JSString* string, StringView input, int startOffset, int** ovector)
{
    ASSERT(owner);
    VM& vm = owner->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    int position = regExp->match(owner, input, startOffset, m_ovector);
    RETURN_IF_EXCEPTION(scope, MatchResult::failed());

    if (ovector)
        *ovector = m_ovector.data();

    if (position == -1)
        return MatchResult::failed();

    ASSERT(!m_ovector.isEmpty());
    ASSERT(m_ovector[0] == position);
    ASSERT(m_ovector[1] >= position);
    size_t end = m_ovector[1];

    m_cachedResult.record(vm, owner, regExp, string, MatchResult(position, end), /* oneCharacterMatch */ false);

    return MatchResult(position, end);
}

ALWAYS_INLINE MatchResult RegExpGlobalData::performMatch(JSGlobalObject* owner, RegExp* regExp, JSString* string, StringView input, int startOffset)
{
    ASSERT(owner);
    VM& vm = owner->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    MatchResult result = regExp->match(owner, input, startOffset);
    RETURN_IF_EXCEPTION(scope, MatchResult::failed());
    if (result)
        m_cachedResult.record(vm, owner, regExp, string, result, /* oneCharacterMatch */ false);
    return result;
}

ALWAYS_INLINE void RegExpGlobalData::recordMatch(VM& vm, JSGlobalObject* owner, RegExp* regExp, JSString* string, const MatchResult& result, bool oneCharacterMatch)
{
    ASSERT(result);
    m_cachedResult.record(vm, owner, regExp, string, result, oneCharacterMatch);
}

inline MatchResult RegExpGlobalData::matchResult() const
{
    return m_cachedResult.result();
}

inline void RegExpGlobalData::resetResultFromCache(JSGlobalObject* owner, RegExp* regExp, JSString* string, MatchResult matchResult, Vector<int>&& vector)
{
    m_ovector = WTFMove(vector);
    m_cachedResult.record(getVM(owner), owner, regExp, string, matchResult, /* oneCharacterMatch */ false);
}

}
