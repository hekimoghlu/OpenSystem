/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 8, 2024.
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

#include "PropertyAllowlist.h"
#include "SelectorFilter.h"
#include "StyleRule.h"

namespace WebCore {
namespace Style {

enum class MatchBasedOnRuleHash : unsigned {
    None,
    Universal,
    ClassA,
    ClassB,
    ClassC
};

enum class IsStartingStyle : bool { No, Yes };

class RuleData {
public:
    static const unsigned maximumSelectorComponentCount = 8192;

    RuleData(const StyleRule&, unsigned selectorIndex, unsigned selectorListIndex, unsigned position, IsStartingStyle);

    unsigned position() const { return m_position; }

    const StyleRule& styleRule() const { return *m_styleRule; }

    const CSSSelector* selector() const
    { 
        return m_styleRule->selectorList().selectorAt(m_selectorIndex);
    }

#if ENABLE(CSS_SELECTOR_JIT)
    CompiledSelector& compiledSelector() const { return m_styleRule->compiledSelectorForListIndex(m_selectorListIndex); }
#endif
    
    unsigned selectorIndex() const { return m_selectorIndex; }
    unsigned selectorListIndex() const { return m_selectorListIndex; }

    bool canMatchPseudoElement() const { return m_canMatchPseudoElement; }
    MatchBasedOnRuleHash matchBasedOnRuleHash() const { return static_cast<MatchBasedOnRuleHash>(m_matchBasedOnRuleHash); }
    bool containsUncommonAttributeSelector() const { return m_containsUncommonAttributeSelector; }
    unsigned linkMatchType() const { return m_linkMatchType; }
    PropertyAllowlist propertyAllowlist() const { return static_cast<PropertyAllowlist>(m_propertyAllowlist); }
    IsStartingStyle isStartingStyle() const { return static_cast<IsStartingStyle>(m_isStartingStyle); }
    bool isEnabled() const { return m_isEnabled; }
    void setEnabled(bool value) { m_isEnabled = value; }

    const SelectorFilter::Hashes& descendantSelectorIdentifierHashes() const { return m_descendantSelectorIdentifierHashes; }

    void disableSelectorFiltering() { m_descendantSelectorIdentifierHashes[0] = 0; }

private:
    RefPtr<const StyleRule> m_styleRule;
    // Keep in sync with RuleFeature's selectorIndex and selectorListIndex size.
    unsigned m_selectorIndex : 16;
    unsigned m_selectorListIndex : 16;
    // If we have more rules than 2^bitcount here we'll get confused about rule order.
    unsigned m_position : 21;
    unsigned m_matchBasedOnRuleHash : 3;
    unsigned m_canMatchPseudoElement : 1;
    unsigned m_containsUncommonAttributeSelector : 1;
    unsigned m_linkMatchType : 2; //  SelectorChecker::LinkMatchMask
    unsigned m_propertyAllowlist : 2;
    unsigned m_isStartingStyle : 1;
    unsigned m_isEnabled : 1;
    SelectorFilter::Hashes m_descendantSelectorIdentifierHashes;
};

} // namespace Style
} // namespace WebCore

namespace WTF {

// RuleData is simple enough that initializing to 0 and moving with memcpy will totally work.
template<> struct VectorTraits<WebCore::Style::RuleData> : SimpleClassVectorTraits { };

} // namespace WTF

