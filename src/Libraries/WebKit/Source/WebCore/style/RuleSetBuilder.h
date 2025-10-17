/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 10, 2022.
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

#include "CSSParserEnum.h"
#include "MediaQuery.h"
#include "RuleSet.h"

namespace WebCore {
namespace Style {

class RuleSetBuilder {
public:
    enum class ShrinkToFit : bool { Enable, Disable };
    enum class ShouldResolveNesting : bool { No, Yes };
    RuleSetBuilder(RuleSet&, const MQ::MediaQueryEvaluator&, Resolver* = nullptr, ShrinkToFit = ShrinkToFit::Enable, ShouldResolveNesting = ShouldResolveNesting::No);
    ~RuleSetBuilder();

    void addRulesFromSheet(const StyleSheetContents&, const MQ::MediaQueryList& sheetQuery = { });
    void addStyleRule(const StyleRule&);

private:
    RuleSetBuilder(const MQ::MediaQueryEvaluator&);

    void addStyleRule(StyleRuleWithNesting&);
    void addStyleRule(StyleRuleNestedDeclarations&);
    void addRulesFromSheetContents(const StyleSheetContents&);
    void addChildRules(const Vector<Ref<StyleRuleBase>>&);
    void addChildRule(Ref<StyleRuleBase>);
    void disallowDynamicMediaQueryEvaluationIfNeeded();
    void addStyleRuleWithSelectorList(const CSSSelectorList&, const StyleRule&);

    void registerLayers(const Vector<CascadeLayerName>&);
    void pushCascadeLayer(const CascadeLayerName&);
    void popCascadeLayer(const CascadeLayerName&);
    void updateCascadeLayerPriorities();

    void addMutatingRulesToResolver();
    void updateDynamicMediaQueries();
    void resolveSelectorListWithNesting(StyleRuleWithNesting&);

    struct MediaQueryCollector {
        ~MediaQueryCollector();

        const MQ::MediaQueryEvaluator& evaluator;
        bool collectDynamic { false };

        struct DynamicContext {
            const MQ::MediaQueryList& queries;
            Vector<size_t> affectedRulePositions { };
            UncheckedKeyHashSet<Ref<const StyleRule>> affectedRules { };
        };
        Vector<DynamicContext> dynamicContextStack { };

        Vector<RuleSet::DynamicMediaQueryRules> dynamicMediaQueryRules { };
        OptionSet<MQ::MediaQueryDynamicDependency> allDynamicDependencies { };

        bool pushAndEvaluate(const MQ::MediaQueryList&);
        void pop(const MQ::MediaQueryList&);
        void addRuleIfNeeded(const RuleData&);
    };

    RefPtr<RuleSet> m_ruleSet;
    MediaQueryCollector m_mediaQueryCollector;
    Resolver* m_resolver { nullptr };
    const ShrinkToFit m_shrinkToFit { ShrinkToFit::Enable };

    CascadeLayerName m_resolvedCascadeLayerName;
    UncheckedKeyHashMap<CascadeLayerName, RuleSet::CascadeLayerIdentifier> m_cascadeLayerIdentifierMap;
    RuleSet::CascadeLayerIdentifier m_currentCascadeLayerIdentifier { 0 };
    Vector<const CSSSelectorList*> m_selectorListStack;
    Vector<CSSParserEnum::NestedContextType> m_ancestorStack;
    const ShouldResolveNesting m_builderShouldResolveNesting { ShouldResolveNesting::No };
    bool m_shouldResolveNestingForSheet { false };

    RuleSet::ContainerQueryIdentifier m_currentContainerQueryIdentifier { 0 };
    RuleSet::ScopeRuleIdentifier m_currentScopeIdentifier { 0 };

    IsStartingStyle m_isStartingStyle { IsStartingStyle::No };

    Vector<RuleSet::ResolverMutatingRule> m_collectedResolverMutatingRules;
    bool requiresStaticMediaQueryEvaluation { false };
};

}
}
