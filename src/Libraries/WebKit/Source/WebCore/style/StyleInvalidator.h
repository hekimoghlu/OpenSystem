/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 25, 2022.
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

#include "RuleFeature.h"
#include "RuleSet.h"
#include <wtf/Forward.h>
#include <wtf/HashMap.h>

namespace WebCore {

class Document;
class Element;
class ShadowRoot;
class StyleSheetContents;

namespace MQ {
class MediaQueryEvaluator;
}

namespace Style {

class Scope;

struct InvalidationRuleSet;
struct SelectorMatchingState;

class Invalidator {
public:
    Invalidator(const Vector<Ref<StyleSheetContents>>&, const MQ::MediaQueryEvaluator&);
    Invalidator(const InvalidationRuleSetVector&);

    ~Invalidator();

    bool dirtiesAllStyle() const { return m_dirtiesAllStyle; }
    void invalidateStyle(Document&);
    void invalidateStyle(Scope&);
    void invalidateStyle(ShadowRoot&);
    void invalidateStyle(Element&);

    static void invalidateShadowParts(ShadowRoot&);

    using MatchElementRuleSets = UncheckedKeyHashMap<MatchElement, InvalidationRuleSetVector, IntHash<MatchElement>, WTF::StrongEnumHashTraits<MatchElement>>;
    static void addToMatchElementRuleSets(Invalidator::MatchElementRuleSets&, const InvalidationRuleSet&);
    static void addToMatchElementRuleSetsRespectingNegation(Invalidator::MatchElementRuleSets&, const InvalidationRuleSet&);
    static void invalidateWithMatchElementRuleSets(Element&, const MatchElementRuleSets&);
    static void invalidateAllStyle(Scope&);
    static void invalidateHostAndSlottedStyleIfNeeded(ShadowRoot&);
    static void invalidateWithScopeBreakingHasPseudoClassRuleSet(Element&, const RuleSet*);

private:
    enum class CheckDescendants : bool { No, Yes };
    CheckDescendants invalidateIfNeeded(Element&, SelectorMatchingState*);
    void invalidateStyleForTree(Element&, SelectorMatchingState*);
    void invalidateStyleForDescendants(Element&, SelectorMatchingState*);
    void invalidateInShadowTreeIfNeeded(Element&);
    void invalidateUserAgentParts(ShadowRoot&);
    void invalidateStyleWithMatchElement(Element&, MatchElement);

    struct RuleInformation {
        bool hasSlottedPseudoElementRules { false };
        bool hasHostPseudoClassRules { false };
        bool hasHostPseudoClassRulesMatchingInShadowTree { false };
        bool hasUserAgentPartRules { false };
        bool hasCuePseudoElementRules { false };
        bool hasPartPseudoElementRules { false };
    };
    RuleInformation collectRuleInformation();

    RefPtr<RuleSet> m_ownedRuleSet;
    const InvalidationRuleSetVector m_ruleSets;

    RuleInformation m_ruleInformation;

    bool m_dirtiesAllStyle { false };
};

}
}
