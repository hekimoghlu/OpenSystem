/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 25, 2023.
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
#include <wtf/TZoneMalloc.h>
#include <wtf/text/AtomStringHash.h>

namespace WebCore {

struct CSSSelectorParserContext;

class MutableCSSSelector;
using MutableCSSSelectorList = Vector<std::unique_ptr<MutableCSSSelector>>;

class MutableCSSSelector {
    WTF_MAKE_TZONE_ALLOCATED(MutableCSSSelector);
public:
    enum class Combinator {
        Child,
        DescendantSpace,
        DirectAdjacent,
        IndirectAdjacent
    };

    static std::unique_ptr<MutableCSSSelector> parsePseudoClassSelector(StringView, const CSSSelectorParserContext&);
    static std::unique_ptr<MutableCSSSelector> parsePseudoElementSelector(StringView, const CSSSelectorParserContext&);
    static std::unique_ptr<MutableCSSSelector> parsePagePseudoSelector(StringView);

    MutableCSSSelector();

    // Recursively copy the selector chain.
    MutableCSSSelector(const CSSSelector&);

    explicit MutableCSSSelector(const QualifiedName&);

    ~MutableCSSSelector();

    std::unique_ptr<CSSSelector> releaseSelector() { return WTFMove(m_selector); }
    const CSSSelector* selector() const { return m_selector.get(); };
    CSSSelector* selector() { return m_selector.get(); }

    void setValue(const AtomString& value, bool matchLowerCase = false) { m_selector->setValue(value, matchLowerCase); }
    const AtomString& value() const { return m_selector->value(); }

    void setAttribute(const QualifiedName& value, CSSSelector::AttributeMatchType type) { m_selector->setAttribute(value, type); }

    void setArgument(const AtomString& value) { m_selector->setArgument(value); }
    void setNth(int a, int b) { m_selector->setNth(a, b); }
    void setMatch(CSSSelector::Match value) { m_selector->setMatch(value); }
    void setRelation(CSSSelector::Relation value) { m_selector->setRelation(value); }
    void setForPage() { m_selector->setForPage(); }

    CSSSelector::Match match() const { return m_selector->match(); }
    CSSSelector::PseudoElement pseudoElement() const { return m_selector->pseudoElement(); }
    const CSSSelectorList* selectorList() const { return m_selector->selectorList(); }

    void setPseudoElement(CSSSelector::PseudoElement type) { m_selector->setPseudoElement(type); }
    void setPseudoClass(CSSSelector::PseudoClass type) { m_selector->setPseudoClass(type); }

    void adoptSelectorVector(MutableCSSSelectorList&&);
    void setArgumentList(FixedVector<AtomString>);
    void setLangList(FixedVector<PossiblyQuotedIdentifier>);
    void setSelectorList(std::unique_ptr<CSSSelectorList>);

    void setImplicit() { m_selector->setImplicit(); }

    CSSSelector::PseudoClass pseudoClass() const { return m_selector->pseudoClass(); }

    bool matchesPseudoElement() const;

    bool isHostPseudoClass() const { return m_selector->isHostPseudoClass(); }

    bool hasExplicitNestingParent() const;
    bool hasExplicitPseudoClassScope() const;

    // FIXME-NEWPARSER: "slotted" was removed here for now, since it leads to a combinator
    // connection of ShadowDescendant, and the current shadow DOM code doesn't expect this. When
    // we do fix this issue, make sure to patch the namespace prependTag code to remove the slotted
    // special case, since it will be covered by this function once again.
    bool needsImplicitShadowCombinatorForMatching() const;

    MutableCSSSelector* tagHistory() const { return m_tagHistory.get(); }
    MutableCSSSelector* leftmostSimpleSelector();
    const MutableCSSSelector* leftmostSimpleSelector() const;
    bool startsWithExplicitCombinator() const;
    void setTagHistory(std::unique_ptr<MutableCSSSelector> selector) { m_tagHistory = WTFMove(selector); }
    void clearTagHistory() { m_tagHistory.reset(); }
    void insertTagHistory(CSSSelector::Relation before, std::unique_ptr<MutableCSSSelector>, CSSSelector::Relation after);
    void appendTagHistory(CSSSelector::Relation, std::unique_ptr<MutableCSSSelector>);
    void appendTagHistory(Combinator, std::unique_ptr<MutableCSSSelector>);
    void appendTagHistoryAsRelative(std::unique_ptr<MutableCSSSelector>);
    void prependTagSelector(const QualifiedName&, bool tagIsForNamespaceRule = false);
    std::unique_ptr<MutableCSSSelector> releaseTagHistory();

private:
    std::unique_ptr<CSSSelector> m_selector;
    std::unique_ptr<MutableCSSSelector> m_tagHistory;
};

// FIXME: WebKitUnknown is listed below as otherwise @supports does the wrong thing, but there ought
// to be a better solution.
inline bool MutableCSSSelector::needsImplicitShadowCombinatorForMatching() const
{
    return match() == CSSSelector::Match::PseudoElement
        && (pseudoElement() == CSSSelector::PseudoElement::UserAgentPart
#if ENABLE(VIDEO)
            || pseudoElement() == CSSSelector::PseudoElement::Cue
#endif
            || pseudoElement() == CSSSelector::PseudoElement::Part
            || pseudoElement() == CSSSelector::PseudoElement::Slotted
            || pseudoElement() == CSSSelector::PseudoElement::UserAgentPartLegacyAlias
            || pseudoElement() == CSSSelector::PseudoElement::WebKitUnknown);
}

} // namespace WebCore
