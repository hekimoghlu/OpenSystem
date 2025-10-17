/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 8, 2024.
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
#include "RenderStyle.h"
#include "Timer.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakRef.h>

namespace WebCore {

namespace Style {

class Resolver;

class MatchedDeclarationsCache {
    WTF_MAKE_TZONE_ALLOCATED(MatchedDeclarationsCache);
public:
    explicit MatchedDeclarationsCache(const Resolver&);
    ~MatchedDeclarationsCache();

    static bool isCacheable(const Element&, const RenderStyle&, const RenderStyle& parentStyle);
    static unsigned computeHash(const MatchResult&, const StyleCustomPropertyData& inheritedCustomProperties);

    struct Entry {
        MatchResult matchResult;
        std::unique_ptr<const RenderStyle> renderStyle;
        std::unique_ptr<const RenderStyle> parentRenderStyle;
        std::unique_ptr<const RenderStyle> userAgentAppearanceStyle;

        bool isUsableAfterHighPriorityProperties(const RenderStyle&) const;
    };

    const Entry* find(unsigned hash, const MatchResult&, const StyleCustomPropertyData& inheritedCustomProperties);
    void add(const RenderStyle&, const RenderStyle& parentStyle, const RenderStyle* userAgentAppearanceStyle, unsigned hash, const MatchResult&);
    void remove(unsigned hash);

    // Every N additions to the matched declaration cache trigger a sweep where entries holding
    // the last reference to a style declaration are garbage collected.
    void invalidate();
    void clearEntriesAffectedByViewportUnits();

    void ref() const;
    void deref() const;

private:
    void sweep();

    SingleThreadWeakRef<const Resolver> m_owner;
    UncheckedKeyHashMap<unsigned, Entry, AlreadyHashed> m_entries;
    Timer m_sweepTimer;
    unsigned m_additionsSinceLastSweep { 0 };
};

}
}
