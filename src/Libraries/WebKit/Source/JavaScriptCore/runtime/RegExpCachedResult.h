/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 25, 2023.
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

#include "MatchResult.h"
#include "SlotVisitorMacros.h"
#include "WriteBarrier.h"

namespace JSC {

class JSArray;
class JSString;
class RegExp;

// RegExpCachedResult is used to track the cached results of the last
// match, stores on the RegExp constructor (e.g. $&, $_, $1, $2 ...).
// These values will be lazily generated on demand, so the cached result
// may be in a lazy or reified state. A lazy state is indicated by a
// value of m_result indicating a successful match, and a reified state
// is indicated by setting m_result to MatchResult::failed().
// Following a successful match, m_result, m_lastInput and m_lastRegExp
// can be used to reify the results from the match, following reification
// m_reifiedResult and m_reifiedInput hold the cached results.
class RegExpCachedResult {
public:
    inline void record(VM&, JSObject* owner, RegExp*, JSString* input, MatchResult, bool oneCharacterMatch);

    JSArray* lastResult(JSGlobalObject*, JSObject* owner);
    void setInput(JSGlobalObject*, JSObject* owner, JSString*);

    JSString* leftContext(JSGlobalObject*, JSObject* owner);
    JSString* rightContext(JSGlobalObject*, JSObject* owner);

    JSString* input()
    {
        return m_reified ? m_reifiedInput.get() : m_lastInput.get();
    }

    DECLARE_VISIT_AGGREGATE;

    // m_lastRegExp would be nullptr when RegExpCachedResult is not reified.
    // If we find m_lastRegExp is nullptr, it means this should hold the empty RegExp.
    static constexpr ptrdiff_t offsetOfLastRegExp() { return OBJECT_OFFSETOF(RegExpCachedResult, m_lastRegExp); }
    static constexpr ptrdiff_t offsetOfLastInput() { return OBJECT_OFFSETOF(RegExpCachedResult, m_lastInput); }
    static constexpr ptrdiff_t offsetOfResult() { return OBJECT_OFFSETOF(RegExpCachedResult, m_result); }
    static constexpr ptrdiff_t offsetOfReified() { return OBJECT_OFFSETOF(RegExpCachedResult, m_reified); }
    static constexpr ptrdiff_t offsetOfOneCharacterMatch() { return OBJECT_OFFSETOF(RegExpCachedResult, m_oneCharacterMatch); }

    MatchResult result() const { return m_result; }

private:
    MatchResult m_result { 0, 0 };
    bool m_reified { false };
    bool m_oneCharacterMatch { false };
    WriteBarrier<JSString> m_lastInput;
    WriteBarrier<RegExp> m_lastRegExp;
    WriteBarrier<JSArray> m_reifiedResult;
    WriteBarrier<JSString> m_reifiedInput;
    WriteBarrier<JSString> m_reifiedLeftContext;
    WriteBarrier<JSString> m_reifiedRightContext;
};

} // namespace JSC
