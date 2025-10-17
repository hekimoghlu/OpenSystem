/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 8, 2023.
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
#include "ElementRuleCollector.h"
#include "InspectorCSSOMWrappers.h"
#include "MatchedDeclarationsCache.h"
#include "MediaQueryEvaluator.h"
#include "RuleSet.h"
#include "StyleScopeRuleSets.h"
#include <memory>
#include <wtf/HashMap.h>
#include <wtf/RefPtr.h>
#include <wtf/Vector.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/AtomStringHash.h>
#include <wtf/text/StringHash.h>

namespace WebCore {

class BlendingKeyframe;
class BlendingKeyframes;
class CSSStyleSheet;
class Document;
class Element;
class RenderStyle;
class RuleData;
class RuleSet;
class SelectorFilter;
class Settings;
class StyleRuleKeyframe;
class StyleProperties;
class StyleRule;
class StyleRuleKeyframes;
class StyleRulePage;
class StyleSheet;
class StyleSheetList;
class ViewportStyleResolver;

// MatchOnlyUserAgentRules is used in media queries, where relative units
// are interpreted according to the document root element style, and styled only
// from the User Agent Stylesheet rules.

enum class RuleMatchingBehavior: uint8_t {
    MatchAllRules,
    MatchAllRulesExcludingSMIL,
    MatchOnlyUserAgentRules,
};

namespace Style {

struct BuilderContext;
struct ResolvedStyle;
struct SelectorMatchingState;

struct ResolutionContext {
    const RenderStyle* parentStyle;
    const RenderStyle* parentBoxStyle { nullptr };
    // This needs to be provided during style resolution when up-to-date document element style is not available via DOM.
    const RenderStyle* documentElementStyle { nullptr };
    SelectorMatchingState* selectorMatchingState { nullptr };
    bool isSVGUseTreeRoot { false };
};

using KeyframesRuleMap = UncheckedKeyHashMap<AtomString, RefPtr<StyleRuleKeyframes>>;

class Resolver : public RefCounted<Resolver>, public CanMakeSingleThreadWeakPtr<Resolver> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(Resolver);
public:
    // Style resolvers are shared between shadow trees with identical styles. That's why we don't simply provide a Style::Scope.
    enum class ScopeType : bool { Document, ShadowTree };
    static Ref<Resolver> create(Document&, ScopeType);
    ~Resolver();

    ResolvedStyle styleForElement(Element&, const ResolutionContext&, RuleMatchingBehavior = RuleMatchingBehavior::MatchAllRules);
    ResolvedStyle styleForElementWithCachedMatchResult(Element&, const ResolutionContext&, const MatchResult&, const RenderStyle& existingRenderStyle);

    void keyframeStylesForAnimation(Element&, const RenderStyle& elementStyle, const ResolutionContext&, BlendingKeyframes&);

    WEBCORE_EXPORT std::optional<ResolvedStyle> styleForPseudoElement(Element&, const PseudoElementRequest&, const ResolutionContext&);

    std::unique_ptr<RenderStyle> styleForPage(int pageIndex);
    std::unique_ptr<RenderStyle> defaultStyleForElement(const Element*);

    Document& document();
    const Document& document() const;
    const Settings& settings() const;

    ScopeType scopeType() const { return m_scopeType; }

    void appendAuthorStyleSheets(const Vector<RefPtr<CSSStyleSheet>>&);

    ScopeRuleSets& ruleSets() { return m_ruleSets; }
    const ScopeRuleSets& ruleSets() const { return m_ruleSets; }

    const MQ::MediaQueryEvaluator& mediaQueryEvaluator() const { return m_mediaQueryEvaluator; }

    void addCurrentSVGFontFaceRules();

    std::unique_ptr<RenderStyle> styleForKeyframe(Element&, const RenderStyle& elementStyle, const ResolutionContext&, const StyleRuleKeyframe&, BlendingKeyframe&);
    bool isAnimationNameValid(const String&);

    void setViewTransitionStyles(CSSSelector::PseudoElement, const AtomString&, Ref<MutableStyleProperties>);

    // These methods will give back the set of rules that matched for a given element (or a pseudo-element).
    enum CSSRuleFilter {
        UAAndUserCSSRules   = 1 << 1,
        AuthorCSSRules      = 1 << 2,
        EmptyCSSRules       = 1 << 3,
        AllButEmptyCSSRules = UAAndUserCSSRules | AuthorCSSRules,
        AllCSSRules         = AllButEmptyCSSRules | EmptyCSSRules,
    };
    Vector<RefPtr<const StyleRule>> styleRulesForElement(const Element*, unsigned rulesToInclude = AllButEmptyCSSRules);
    Vector<RefPtr<const StyleRule>> pseudoStyleRulesForElement(const Element*, const std::optional<Style::PseudoElementRequest>&, unsigned rulesToInclude = AllButEmptyCSSRules);

    bool hasSelectorForId(const AtomString&) const;
    bool hasSelectorForAttribute(const Element&, const AtomString&) const;

    bool hasViewportDependentMediaQueries() const;
    std::optional<DynamicMediaQueryEvaluationChanges> evaluateDynamicMediaQueries();

    static KeyframesRuleMap& userAgentKeyframes();
    static void addUserAgentKeyframeStyle(Ref<StyleRuleKeyframes>&&);
    void addKeyframeStyle(Ref<StyleRuleKeyframes>&&);
    Vector<Ref<StyleRuleKeyframe>> keyframeRulesForName(const AtomString&) const;

    RefPtr<StyleRuleViewTransition> viewTransitionRule() const;

    bool usesFirstLineRules() const { return m_ruleSets.features().usesFirstLineRules; }
    bool usesFirstLetterRules() const { return m_ruleSets.features().usesFirstLetterRules; }
    bool usesStartingStyleRules() const { return m_ruleSets.features().hasStartingStyleRules; }

    void invalidateMatchedDeclarationsCache();
    void clearCachedDeclarationsAffectedByViewportUnits();

    InspectorCSSOMWrappers& inspectorCSSOMWrappers() { return m_inspectorCSSOMWrappers; }

    bool isSharedBetweenShadowTrees() const { return m_isSharedBetweenShadowTrees; }
    void setSharedBetweenShadowTrees() { m_isSharedBetweenShadowTrees = true; }

    const RenderStyle* rootDefaultStyle() const { return m_rootDefaultStyle.get(); }

private:
    Resolver(Document&, ScopeType);
    void initialize();

    class State;

    State initializeStateAndStyle(const Element&, const ResolutionContext&);
    BuilderContext builderContext(const State&);

    void applyMatchedProperties(State&, const MatchResult&);

    WeakPtr<Document, WeakPtrImplWithEventTargetData> m_document;
    const ScopeType m_scopeType;

    ScopeRuleSets m_ruleSets;

    KeyframesRuleMap m_keyframesRuleMap;

    MQ::MediaQueryEvaluator m_mediaQueryEvaluator;
    std::unique_ptr<RenderStyle> m_rootDefaultStyle;

    InspectorCSSOMWrappers m_inspectorCSSOMWrappers;

    MatchedDeclarationsCache m_matchedDeclarationsCache;

    bool m_matchAuthorAndUserStyles { true };
    bool m_isSharedBetweenShadowTrees { false };
};

} // namespace Style
} // namespace WebCore
