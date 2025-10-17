/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 8, 2023.
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
#include "RuleData.h"

#include "CSSFontSelector.h"
#include "CSSKeyframesRule.h"
#include "CSSSelector.h"
#include "CSSSelectorList.h"
#include "CommonAtomStrings.h"
#include "DocumentInlines.h"
#include "HTMLNames.h"
#include "ScriptExecutionContext.h"
#include "SecurityOrigin.h"
#include "SelectorChecker.h"
#include "SelectorFilter.h"
#include "StyleResolver.h"
#include "StyleRule.h"
#include "StyleRuleImport.h"
#include "StyleSheetContents.h"
#include "UserAgentParts.h"

namespace WebCore {
namespace Style {

using namespace HTMLNames;

struct SameSizeAsRuleData {
    void* a;
    unsigned b;
    unsigned c;
    unsigned d[4];
};

static_assert(sizeof(RuleData) == sizeof(SameSizeAsRuleData), "RuleData should stay small");

static inline MatchBasedOnRuleHash computeMatchBasedOnRuleHash(const CSSSelector& selector)
{
    if (selector.tagHistory())
        return MatchBasedOnRuleHash::None;

    if (selector.match() == CSSSelector::Match::Tag) {
        const QualifiedName& tagQualifiedName = selector.tagQName();
        const AtomString& selectorNamespace = tagQualifiedName.namespaceURI();
        if (selectorNamespace == starAtom() || selectorNamespace == xhtmlNamespaceURI) {
            if (tagQualifiedName == anyQName())
                return MatchBasedOnRuleHash::Universal;
            return MatchBasedOnRuleHash::ClassC;
        }
        return MatchBasedOnRuleHash::None;
    }
    if (SelectorChecker::isCommonPseudoClassSelector(&selector))
        return MatchBasedOnRuleHash::ClassB;
    if (selector.match() == CSSSelector::Match::Id)
        return MatchBasedOnRuleHash::ClassA;
    if (selector.match() == CSSSelector::Match::Class)
        return MatchBasedOnRuleHash::ClassB;
    // FIXME: Valueless [attribute] case can be handled here too.

    return MatchBasedOnRuleHash::None;
}

static bool selectorCanMatchPseudoElement(const CSSSelector& rootSelector)
{
    const CSSSelector* selector = &rootSelector;
    do {
        if (selector->matchesPseudoElement())
            return true;

        if (const CSSSelectorList* selectorList = selector->selectorList()) {
            for (const CSSSelector* subSelector = selectorList->first(); subSelector; subSelector = CSSSelectorList::next(subSelector)) {
                if (selectorCanMatchPseudoElement(*subSelector))
                    return true;
            }
        }

        selector = selector->tagHistory();
    } while (selector);
    return false;
}

static inline bool isCommonAttributeSelectorAttribute(const QualifiedName& attribute)
{
    // These are explicitly tested for equality in canShareStyleWithElement.
    return attribute == typeAttr || attribute == readonlyAttr;
}

static bool computeContainsUncommonAttributeSelector(const CSSSelector& rootSelector, bool matchesRightmostElement = true)
{
    const CSSSelector* selector = &rootSelector;
    do {
        if (selector->isAttributeSelector()) {
            // FIXME: considering non-rightmost simple selectors is necessary because of the style sharing of cousins.
            // It is a primitive solution which disable a lot of style sharing on pages that rely on attributes for styling.
            // We should investigate better ways of doing this.
            if (!isCommonAttributeSelectorAttribute(selector->attribute()) || !matchesRightmostElement)
                return true;
        }

        if (const CSSSelectorList* selectorList = selector->selectorList()) {
            for (const CSSSelector* subSelector = selectorList->first(); subSelector; subSelector = CSSSelectorList::next(subSelector)) {
                if (computeContainsUncommonAttributeSelector(*subSelector, matchesRightmostElement))
                    return true;
            }
        }

        if (selector->relation() != CSSSelector::Relation::Subselector)
            matchesRightmostElement = false;

        selector = selector->tagHistory();
    } while (selector);
    return false;
}

static inline PropertyAllowlist determinePropertyAllowlist(const CSSSelector* selector)
{
    for (const CSSSelector* component = selector; component; component = component->tagHistory()) {
#if ENABLE(VIDEO)
        // Property allow-list for `::cue`:
        if (component->match() == CSSSelector::Match::PseudoElement && component->pseudoElement() == CSSSelector::PseudoElement::UserAgentPart && component->value() == UserAgentParts::cue())
            return PropertyAllowlist::Cue;
        // Property allow-list for `::cue(selector)`:
        if (component->match() == CSSSelector::Match::PseudoElement && component->pseudoElement() == CSSSelector::PseudoElement::Cue)
            return PropertyAllowlist::CueSelector;
        // Property allow-list for '::-internal-cue-background':
        if (component->match() == CSSSelector::Match::PseudoElement && component->pseudoElement() == CSSSelector::PseudoElement::UserAgentPart && component->value() == UserAgentParts::internalCueBackground())
            return PropertyAllowlist::CueBackground;
#endif
        if (component->match() == CSSSelector::Match::PseudoElement && component->pseudoElement() == CSSSelector::PseudoElement::Marker)
            return propertyAllowlistForPseudoId(PseudoId::Marker);

        if (const auto* selectorList = selector->selectorList()) {
            for (const auto* subSelector = selectorList->first(); subSelector; subSelector = CSSSelectorList::next(subSelector)) {
                auto allowlistType = determinePropertyAllowlist(subSelector);
                if (allowlistType != PropertyAllowlist::None)
                    return allowlistType;
            }
        }
    }
    return PropertyAllowlist::None;
}

RuleData::RuleData(const StyleRule& styleRule, unsigned selectorIndex, unsigned selectorListIndex, unsigned position, IsStartingStyle isStartingStyle)
    : m_styleRule(&styleRule)
    , m_selectorIndex(selectorIndex)
    , m_selectorListIndex(selectorListIndex)
    , m_position(position)
    , m_matchBasedOnRuleHash(enumToUnderlyingType(computeMatchBasedOnRuleHash(*selector())))
    , m_canMatchPseudoElement(selectorCanMatchPseudoElement(*selector()))
    , m_containsUncommonAttributeSelector(computeContainsUncommonAttributeSelector(*selector()))
    , m_linkMatchType(SelectorChecker::determineLinkMatchType(selector()))
    , m_propertyAllowlist(enumToUnderlyingType(determinePropertyAllowlist(selector())))
    , m_isStartingStyle(enumToUnderlyingType(isStartingStyle))
    , m_isEnabled(true)
    , m_descendantSelectorIdentifierHashes(SelectorFilter::collectHashes(*selector()))
{
    ASSERT(m_position == position);
    ASSERT(m_selectorIndex == selectorIndex);
}

} // namespace Style
} // namespace WebCore
