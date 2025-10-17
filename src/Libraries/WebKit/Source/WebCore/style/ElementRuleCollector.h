/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 22, 2022.
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
#include "MediaQueryEvaluator.h"
#include "PropertyAllowlist.h"
#include "PseudoElementRequest.h"
#include "RuleSet.h"
#include "SelectorChecker.h"
#include "StyleScopeOrdinal.h"
#include <memory>
#include <wtf/RefPtr.h>
#include <wtf/Vector.h>

namespace WebCore::Style {

class ScopeRuleSets;
struct MatchRequest;
struct SelectorMatchingState;
enum class CascadeLevel : uint8_t;

struct MatchedRule {
    const RuleData* ruleData { nullptr };
    unsigned specificity { 0 };
    unsigned scopingRootDistance { 0 };
    ScopeOrdinal styleScopeOrdinal;
    CascadeLayerPriority cascadeLayerPriority;
};

class ElementRuleCollector {
public:
    ElementRuleCollector(const Element&, const ScopeRuleSets&, SelectorMatchingState*);
    ElementRuleCollector(const Element&, const RuleSet& authorStyle, SelectorMatchingState*);

    void setIncludeEmptyRules(bool value) { m_shouldIncludeEmptyRules = value; }

    void matchAllRules(bool matchAuthorAndUserStyles, bool includeSMILProperties);
    void matchUARules();
    void matchAuthorRules();
    void matchUserRules();

    bool matchesAnyAuthorRules();
    bool matchesAnyRules(const RuleSet&);

    void setMode(SelectorChecker::Mode mode) { m_mode = mode; }

    void setPseudoElementRequest(const std::optional<PseudoElementRequest>& request) { m_pseudoElementRequest = request; }
    void setMedium(const MQ::MediaQueryEvaluator& medium) { m_isPrintStyle = medium.isPrintMedia(); }


    const MatchResult& matchResult() const;
    std::unique_ptr<MatchResult> releaseMatchResult();

    const Vector<RefPtr<const StyleRule>>& matchedRuleList() const;

    void clearMatchedRules();

    const PseudoIdSet& matchedPseudoElementIds() const { return m_matchedPseudoElementIds; }
    const Relations& styleRelations() const { return m_styleRelations; }
    bool didMatchUncommonAttributeSelector() const { return m_didMatchUncommonAttributeSelector; }

    void addAuthorKeyframeRules(const StyleRuleKeyframe&);

private:
    void addElementStyleProperties(const StyleProperties*, CascadeLayerPriority, IsCacheable = IsCacheable::Yes, FromStyleAttribute = FromStyleAttribute::No);

    void matchUARules(const RuleSet&);

    void addElementInlineStyleProperties(bool includeSMILProperties);

    void matchUserAgentPartRules(CascadeLevel);
    void matchHostPseudoClassRules(CascadeLevel);
    void matchSlottedPseudoElementRules(CascadeLevel);
    void matchPartPseudoElementRules(CascadeLevel);
    void matchPartPseudoElementRulesForScope(const Element& partMatchingElement, CascadeLevel);

    void collectMatchingUserAgentPartRules(const MatchRequest&);

    void collectMatchingRules(CascadeLevel);
    void collectMatchingRules(const MatchRequest&);
    void collectMatchingRulesForList(const RuleSet::RuleDataVector*, const MatchRequest&);
    bool isFirstMatchModeAndHasMatchedAnyRules() const;
    bool ruleMatches(const RuleData&, unsigned& specificity, ScopeOrdinal, const ContainerNode* scopingRoot = nullptr);
    bool containerQueriesMatch(const RuleData&, const MatchRequest&);
    struct ScopingRootWithDistance {
        RefPtr<const ContainerNode> scopingRoot;
        unsigned distance { std::numeric_limits<unsigned>::max() };
    };
    std::pair<bool, std::optional<Vector<ScopingRootWithDistance>>> scopeRulesMatch(const RuleData&, const MatchRequest&);

    void sortMatchedRules();

    enum class DeclarationOrigin { UserAgent, User, Author };
    Vector<MatchedProperties>& declarationsForOrigin(DeclarationOrigin);
    void sortAndTransferMatchedRules(DeclarationOrigin);
    void transferMatchedRules(DeclarationOrigin, std::optional<ScopeOrdinal> forScope = { });

    void addMatchedRule(const RuleData&, unsigned specificity, unsigned scopingRootDistance, const MatchRequest&);
    void addMatchedProperties(MatchedProperties&&, DeclarationOrigin);

    const Element& element() const { return m_element.get(); }

    const Ref<const Element> m_element;
    Ref<const RuleSet> m_authorStyle;
    RefPtr<const RuleSet> m_userStyle;
    RefPtr<const RuleSet> m_userAgentMediaQueryStyle;
    RefPtr<const RuleSet> m_dynamicViewTransitionsStyle;
    SelectorMatchingState* m_selectorMatchingState;

    bool m_shouldIncludeEmptyRules { false };
    bool m_isPrintStyle { false };
    // FIXME: This should be a SelectorChecker::Mode.
    bool m_firstMatchMode { false };
    std::optional<PseudoElementRequest> m_pseudoElementRequest { };
    SelectorChecker::Mode m_mode { SelectorChecker::Mode::ResolvingStyle };

    Vector<MatchedRule, 64> m_matchedRules;
    size_t m_matchedRuleTransferIndex { 0 };

    // Output.
    Vector<RefPtr<const StyleRule>> m_matchedRuleList;
    bool m_didMatchUncommonAttributeSelector { false };
    std::unique_ptr<MatchResult> m_result;
    Relations m_styleRelations;
    PseudoIdSet m_matchedPseudoElementIds;
};

}
