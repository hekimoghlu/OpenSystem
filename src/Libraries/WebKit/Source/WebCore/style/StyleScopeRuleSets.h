/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 4, 2025.
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
#include "UserAgentStyle.h"
#include <memory>
#include <wtf/HashMap.h>
#include <wtf/RefPtr.h>
#include <wtf/Vector.h>

namespace WebCore {

class CSSStyleRule;
class CSSStyleSheet;
class ExtensionStyleSheets;

namespace MQ {
class MediaQueryEvaluator;
};

namespace Style {

enum class CascadeLevel : uint8_t;
class InspectorCSSOMWrappers;
class Resolver;

struct InvalidationRuleSet {
    RefPtr<RuleSet> ruleSet;
    Vector<const CSSSelector*> invalidationSelectors;
    MatchElement matchElement;
    IsNegation isNegation;
};

enum class SelectorsForStyleAttribute : uint8_t { None, SubjectPositionOnly, NonSubjectPosition };

class ScopeRuleSets {
public:
    ScopeRuleSets(Resolver&);
    ~ScopeRuleSets();

    bool isAuthorStyleDefined() const { return m_isAuthorStyleDefined; }
    RuleSet* userAgentMediaQueryStyle() const;
    RuleSet* dynamicViewTransitionsStyle() const;
    RuleSet& authorStyle() const { return *m_authorStyle; }
    RuleSet* userStyle() const;
    RuleSet* styleForCascadeLevel(CascadeLevel);

    const RuleFeatureSet& features() const;
    RuleSet* sibling() const { return m_siblingRuleSet.get(); }
    RuleSet* uncommonAttribute() const { return m_uncommonAttributeRuleSet.get(); }
    RuleSet* scopeBreakingHasPseudoClassInvalidationRuleSet() const { return m_scopeBreakingHasPseudoClassInvalidationRuleSet.get(); }

    const Vector<InvalidationRuleSet>* idInvalidationRuleSets(const AtomString&) const;
    const Vector<InvalidationRuleSet>* classInvalidationRuleSets(const AtomString&) const;
    const Vector<InvalidationRuleSet>* attributeInvalidationRuleSets(const AtomString&) const;
    const Vector<InvalidationRuleSet>* pseudoClassInvalidationRuleSets(const PseudoClassInvalidationKey&) const;
    const Vector<InvalidationRuleSet>* hasPseudoClassInvalidationRuleSets(const PseudoClassInvalidationKey&) const;

    const UncheckedKeyHashSet<AtomString>& customPropertyNamesInStyleContainerQueries() const;

    SelectorsForStyleAttribute selectorsForStyleAttribute() const;

    void setUsesSharedUserStyle(bool b) { m_usesSharedUserStyle = b; }
    void initializeUserStyle();

    void resetAuthorStyle();
    void appendAuthorStyleSheets(const Vector<RefPtr<CSSStyleSheet>>&, MQ::MediaQueryEvaluator*, Style::InspectorCSSOMWrappers&);

    void resetUserAgentMediaQueryStyle();

    bool hasViewportDependentMediaQueries() const;
    bool hasContainerQueries() const;
    bool hasScopeRules() const;

    RefPtr<StyleRuleViewTransition> viewTransitionRule() const;

    std::optional<DynamicMediaQueryEvaluationChanges> evaluateDynamicMediaQueryRules(const MQ::MediaQueryEvaluator&);

    RuleFeatureSet& mutableFeatures();

    void setDynamicViewTransitionsStyle(RuleSet* ruleSet)
    {
        m_dynamicViewTransitionsStyle = ruleSet;
    }

    bool& isInvalidatingStyleWithRuleSets() { return m_isInvalidatingStyleWithRuleSets; }

    bool hasMatchingUserOrAuthorStyle(const WTF::Function<bool(RuleSet&)>&);

private:
    void collectFeatures() const;
    void collectRulesFromUserStyleSheets(const Vector<RefPtr<CSSStyleSheet>>&, RuleSet& userStyle, const MQ::MediaQueryEvaluator&);
    void updateUserAgentMediaQueryStyleIfNeeded() const;

    RefPtr<RuleSet> m_authorStyle;
    mutable RefPtr<RuleSet> m_userAgentMediaQueryStyle;
    mutable RefPtr<RuleSet> m_dynamicViewTransitionsStyle;
    RefPtr<RuleSet> m_userStyle;

    Resolver& m_styleResolver;
    mutable RuleFeatureSet m_features;
    mutable RefPtr<RuleSet> m_siblingRuleSet;
    mutable RefPtr<RuleSet> m_uncommonAttributeRuleSet;
    mutable RefPtr<RuleSet> m_scopeBreakingHasPseudoClassInvalidationRuleSet;
    mutable UncheckedKeyHashMap<AtomString, std::unique_ptr<Vector<InvalidationRuleSet>>> m_idInvalidationRuleSets;
    mutable UncheckedKeyHashMap<AtomString, std::unique_ptr<Vector<InvalidationRuleSet>>> m_classInvalidationRuleSets;
    mutable UncheckedKeyHashMap<AtomString, std::unique_ptr<Vector<InvalidationRuleSet>>> m_attributeInvalidationRuleSets;
    mutable UncheckedKeyHashMap<PseudoClassInvalidationKey, std::unique_ptr<Vector<InvalidationRuleSet>>> m_pseudoClassInvalidationRuleSets;
    mutable UncheckedKeyHashMap<PseudoClassInvalidationKey, std::unique_ptr<Vector<InvalidationRuleSet>>> m_hasPseudoClassInvalidationRuleSets;

    mutable std::optional<UncheckedKeyHashSet<AtomString>> m_customPropertyNamesInStyleContainerQueries;

    mutable std::optional<SelectorsForStyleAttribute> m_cachedSelectorsForStyleAttribute;

    mutable unsigned m_defaultStyleVersionOnFeatureCollection { 0 };
    mutable unsigned m_userAgentMediaQueryRuleCountOnUpdate { 0 };

    bool m_usesSharedUserStyle { false };
    bool m_isAuthorStyleDefined { false };

    // For catching <rdar://problem/53413013>
    bool m_isInvalidatingStyleWithRuleSets { false };
};

inline const RuleFeatureSet& ScopeRuleSets::features() const
{
    if (m_defaultStyleVersionOnFeatureCollection < UserAgentStyle::defaultStyleVersion)
        collectFeatures();
    return m_features;
}

// FIXME: There should be just the const version.
inline RuleFeatureSet& ScopeRuleSets::mutableFeatures()
{
    if (m_defaultStyleVersionOnFeatureCollection < UserAgentStyle::defaultStyleVersion)
        collectFeatures();
    return m_features;
}

} // namespace Style
} // namespace WebCore
