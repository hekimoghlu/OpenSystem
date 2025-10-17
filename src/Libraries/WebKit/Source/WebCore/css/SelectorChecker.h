/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 7, 2024.
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

#include "CSSSelector.h"
#include "Element.h"
#include "SelectorMatchingState.h"
#include "StyleRelations.h"
#include "StyleScopeOrdinal.h"
#include "StyleScrollbarState.h"

namespace WebCore {

class CSSSelector;
class Element;
class RenderScrollbar;
class RenderStyle;

class SelectorChecker {
    WTF_MAKE_NONCOPYABLE(SelectorChecker);
    enum class Match { SelectorMatches, SelectorFailsLocally, SelectorFailsAllSiblings, SelectorFailsCompletely };

    enum class MatchType { VirtualPseudoElementOnly, Element };

    struct MatchResult {
        Match match;
        MatchType matchType;

        static MatchResult matches(MatchType matchType)
        {
            return { Match::SelectorMatches, matchType };
        }

        static MatchResult updateWithMatchType(MatchResult result, MatchType matchType)
        {
            if (matchType == MatchType::VirtualPseudoElementOnly)
                result.matchType = MatchType::VirtualPseudoElementOnly;
            return result;
        }

        static MatchResult fails(Match match)
        {
            return { match, MatchType::Element };
        }
    };

public:
    enum class Mode : unsigned char {
        ResolvingStyle = 0, CollectingRules, CollectingRulesIgnoringVirtualPseudoElements, QueryingRules
    };

    SelectorChecker(Document&);

    struct CheckingContext {
        CheckingContext(SelectorChecker::Mode resolvingMode)
            : resolvingMode(resolvingMode)
        { }

        const SelectorChecker::Mode resolvingMode;
        // FIXME: Switch to PseudoElementIdentifier.
        PseudoId pseudoId { PseudoId::None };
        AtomString pseudoElementNameArgument;
        std::optional<StyleScrollbarState> scrollbarState;
        Vector<AtomString> classList;
        RefPtr<const ContainerNode> scope;
        const Element* hasScope { nullptr };
        bool matchesAllHasScopes { false };
        Style::ScopeOrdinal styleScopeOrdinal { Style::ScopeOrdinal::Element };
        Style::SelectorMatchingState* selectorMatchingState { nullptr };

        // FIXME: It would be nicer to have a separate object for return values. This requires some more work in the selector compiler.
        Style::Relations styleRelations;
        PseudoIdSet pseudoIDSet;
        bool matchedInsideScope { false };
        bool disallowHasPseudoClass { false };
    };

    bool match(const CSSSelector&, const Element&, CheckingContext&) const;

    bool matchHostPseudoClass(const CSSSelector&, const Element&, CheckingContext&) const;

    static bool isCommonPseudoClassSelector(const CSSSelector*);
    static bool attributeSelectorMatches(const Element&, const QualifiedName&, const AtomString& attributeValue, const CSSSelector&);

    enum LinkMatchMask { MatchDefault = 0, MatchLink = 1, MatchVisited = 2, MatchAll = MatchLink | MatchVisited };
    static unsigned determineLinkMatchType(const CSSSelector*);

    struct LocalContext;
    
private:
    MatchResult matchRecursively(CheckingContext&, LocalContext&, PseudoIdSet&) const;
    bool checkOne(CheckingContext&, LocalContext&, MatchType&) const;
    bool matchSelectorList(CheckingContext&, const LocalContext&, const Element&, const CSSSelectorList&) const;
    bool matchHasPseudoClass(CheckingContext&, const Element&, const CSSSelector&) const;

    bool checkScrollbarPseudoClass(const CheckingContext&, const Element&, const CSSSelector&) const;
    bool checkViewTransitionPseudoClass(const CheckingContext&, const Element&, const CSSSelector&) const;

    bool m_strictParsing;
    bool m_documentIsHTML;
};

inline bool SelectorChecker::isCommonPseudoClassSelector(const CSSSelector* selector)
{
    if (selector->match() != CSSSelector::Match::PseudoClass)
        return false;
    auto pseudoType = selector->pseudoClass();
    return pseudoType == CSSSelector::PseudoClass::Link
        || pseudoType == CSSSelector::PseudoClass::AnyLink
        || pseudoType == CSSSelector::PseudoClass::Visited
        || pseudoType == CSSSelector::PseudoClass::Focus;
}

} // namespace WebCore
