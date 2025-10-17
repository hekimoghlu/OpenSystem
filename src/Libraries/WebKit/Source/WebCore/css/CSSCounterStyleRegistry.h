/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 8, 2022.
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

#include "CSSCounterStyle.h"
#include <wtf/Forward.h>
#include <wtf/HashMap.h>
#include <wtf/text/AtomStringHash.h>

namespace WebCore {

struct ListStyleType;
class StyleRuleCounterStyle;
enum CSSValueID : uint16_t;

using CounterStyleMap = UncheckedKeyHashMap<AtomString, RefPtr<CSSCounterStyle>>;

DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER(CSSCounterStyleRegistry);
class CSSCounterStyleRegistry {
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(CSSCounterStyleRegistry);
public:
    CSSCounterStyleRegistry() = default;
    RefPtr<CSSCounterStyle> resolvedCounterStyle(const ListStyleType&);
    static RefPtr<CSSCounterStyle> decimalCounter();
    static void addUserAgentCounterStyle(const CSSCounterStyleDescriptors&);
    void addCounterStyle(const CSSCounterStyleDescriptors&);
    static void resolveUserAgentReferences();
    void resolveReferencesIfNeeded();
    bool operator==(const CSSCounterStyleRegistry& other) const;
    void clearAuthorCounterStyles();
    bool hasAuthorCounterStyles() const { return !m_authorCounterStyles.isEmpty(); }

private:
    static CounterStyleMap& userAgentCounterStyles();
    // If no map is passed on, user-agent counter styles map will be used
    static void resolveFallbackReference(CSSCounterStyle&, CounterStyleMap* = nullptr);
    static void resolveExtendsReference(CSSCounterStyle&, CounterStyleMap* = nullptr);
    static void resolveExtendsReference(CSSCounterStyle&, UncheckedKeyHashSet<CSSCounterStyle*>&, CounterStyleMap* = nullptr);
    static RefPtr<CSSCounterStyle> counterStyle(const AtomString&, CounterStyleMap* = nullptr);
    void invalidate();

    CounterStyleMap m_authorCounterStyles;
    bool m_hasUnresolvedReferences { true };
};

} // namespace WebCore
