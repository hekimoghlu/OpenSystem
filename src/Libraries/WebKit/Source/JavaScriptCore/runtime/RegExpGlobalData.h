/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 20, 2024.
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

#include "RegExpCachedResult.h"
#include "RegExpSubstringGlobalAtomCache.h"

namespace JSC {

class JSGlobalObject;

class RegExpGlobalData {
public:
    RegExpCachedResult& cachedResult() { return m_cachedResult; }

    void setMultiline(bool multiline) { m_multiline = multiline; }
    bool multiline() const { return m_multiline; }

    void setInput(JSGlobalObject*, JSString*);
    JSString* input() { return m_cachedResult.input(); }

    DECLARE_VISIT_AGGREGATE;

    JSValue getBackref(JSGlobalObject*, unsigned);
    JSValue getLastParen(JSGlobalObject*);
    JSValue getLeftContext(JSGlobalObject*);
    JSValue getRightContext(JSGlobalObject*);

    MatchResult performMatch(JSGlobalObject*, RegExp*, JSString*, StringView, int startOffset, int** ovector);
    MatchResult performMatch(JSGlobalObject*, RegExp*, JSString*, StringView, int startOffset);
    void recordMatch(VM&, JSGlobalObject*, RegExp*, JSString*, const MatchResult&, bool oneCharacterMatch);

    static constexpr ptrdiff_t offsetOfCachedResult() { return OBJECT_OFFSETOF(RegExpGlobalData, m_cachedResult); }

    const Vector<int>& ovector() const { return m_ovector; }

    inline MatchResult matchResult() const;
    void resetResultFromCache(JSGlobalObject* owner, RegExp*, JSString*, MatchResult, Vector<int>&&);

    RegExpSubstringGlobalAtomCache& substringGlobalAtomCache() { return m_substringGlobalAtomCache; }

private:
    RegExpCachedResult m_cachedResult;
    RegExpSubstringGlobalAtomCache m_substringGlobalAtomCache;
    bool m_multiline { false };
    Vector<int> m_ovector;
};

}
