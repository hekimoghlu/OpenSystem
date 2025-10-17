/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 18, 2025.
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
#include "ContainerQueryEvaluator.h"

#include "CSSPrimitiveValue.h"
#include "CSSToLengthConversionData.h"
#include "CSSValueList.h"
#include "ComposedTreeAncestorIterator.h"
#include "ContainerQueryFeatures.h"
#include "Document.h"
#include "MediaList.h"
#include "NodeRenderStyle.h"
#include "RenderStyleInlines.h"
#include "RenderView.h"
#include "StyleRule.h"
#include "StyleScope.h"

namespace WebCore::Style {

ContainerQueryEvaluator::ContainerQueryEvaluator(const Element& element, SelectionMode selectionMode, ScopeOrdinal scopeOrdinal, ContainerQueryEvaluationState* evaluationState)
    : m_element(element)
    , m_selectionMode(selectionMode)
    , m_scopeOrdinal(scopeOrdinal)
    , m_evaluationState(evaluationState)
{
}

bool ContainerQueryEvaluator::evaluate(const CQ::ContainerQuery& containerQuery) const
{
    auto context = featureEvaluationContextForQuery(containerQuery);
    if (!context)
        return false;

    return evaluateCondition(containerQuery.condition, *context) == MQ::EvaluationResult::True;
}

static const RenderStyle* styleForContainer(const Element& container, OptionSet<CQ::Axis> requiredAxes, const ContainerQueryEvaluationState* evaluationState)
{
    // Any element can be a style container and we haven't necessarily committed the style to render tree yet.
    // Look it up from the currently computed style update instead.
    if (requiredAxes.isEmpty() && evaluationState)
        return evaluationState->styleUpdate->elementStyle(container);

    return container.existingComputedStyle();
}

auto ContainerQueryEvaluator::featureEvaluationContextForQuery(const CQ::ContainerQuery& containerQuery) const -> std::optional<MQ::FeatureEvaluationContext>
{
    // "For each element, the query container to be queried is selected from among the elementâ€™s
    // ancestor query containers that have a valid container-type for all the container features
    // in the <container-condition>. The optional <container-name> filters the set of query containers
    // considered to just those with a matching query container name."
    // https://drafts.csswg.org/css-contain-3/#container-rule

    // "If the <container-query> contains unknown or unsupported container features, no query container will be selected."
    if (containerQuery.containsUnknownFeature == CQ::ContainsUnknownFeature::Yes)
        return { };

    Ref element = m_element;
    RefPtr container = selectContainer(containerQuery.requiredAxes, containerQuery.name, element.get(), m_selectionMode, m_scopeOrdinal, m_evaluationState);
    if (!container)
        return { };

    CheckedPtr containerStyle = styleForContainer(*container.get(), containerQuery.requiredAxes, m_evaluationState);
    if (!containerStyle)
        return { };

    RefPtr containerParent = container->parentElementInComposedTree();
    CheckedPtr containerParentStyle = containerParent ? styleForContainer(*containerParent, containerQuery.requiredAxes, m_evaluationState) : containerStyle;

    Ref document = element->document();
    CheckedPtr rootStyle = document->documentElement()->renderStyle();

    return MQ::FeatureEvaluationContext {
        document.get(),
        CSSToLengthConversionData { *containerStyle, rootStyle.get(), containerParentStyle.get(), document->renderView(), container.get() },
        container->renderer()
    };
}

const Element* ContainerQueryEvaluator::selectContainer(OptionSet<CQ::Axis> requiredAxes, const String& name, const Element& element, SelectionMode selectionMode, ScopeOrdinal scopeOrdinal, const ContainerQueryEvaluationState* evaluationState)
{
    // "For each element, the query container to be queried is selected from among the elementâ€™s
    // ancestor query containers that have a valid container-type for all the container features
    // in the <container-condition>. The optional <container-name> filters the set of query containers
    // considered to just those with a matching query container name."
    // https://drafts.csswg.org/css-contain-3/#container-rule

    auto isValidContainerForRequiredAxes = [&](ContainerType containerType, const RenderElement* principalBox) {
        // Any container is valid for style queries.
        if (requiredAxes.isEmpty())
            return true;

        switch (containerType) {
        case ContainerType::Size:
            return true;
        case ContainerType::InlineSize:
            // Without a principal box the container matches but the query against it will evaluate to Unknown.
            if (!principalBox)
                return true;
            if (requiredAxes.contains(CQ::Axis::Block))
                return false;
            return !requiredAxes.contains(principalBox->isHorizontalWritingMode() ? CQ::Axis::Height : CQ::Axis::Width);
        case ContainerType::Normal:
            return false;
        }
        RELEASE_ASSERT_NOT_REACHED();
    };

    auto isContainerForQuery = [&](const Element& candidateElement, const Element* originatingElement = nullptr) {
        auto style = styleForContainer(candidateElement, requiredAxes, evaluationState);
        if (!style)
            return false;
        if (!isValidContainerForRequiredAxes(style->containerType(), candidateElement.renderer()))
            return false;
        if (name.isEmpty())
            return true;

        return style->containerNames().containsIf([&](auto& scopedName) {
            auto isNameFromAllowedScope = [&](auto& scopedName) {
                // Names from :host rules are allowed when the candidate is the host element.
                RefPtr host = originatingElement ? originatingElement->shadowHost() : element.shadowHost();
                auto isHost = host == &candidateElement;
                if (scopedName.scopeOrdinal == ScopeOrdinal::Shadow && isHost)
                    return true;
                // Otherwise names from the inner scopes are ignored.
                return scopedName.scopeOrdinal <= ScopeOrdinal::Element;
            };
            return isNameFromAllowedScope(scopedName) && scopedName.name == name;
        });
    };

    auto findOriginatingElement = [&]() -> const Element* {
        // ::part() selectors can query its originating host, but not internal query containers inside the shadow tree.
        if (selectionMode == SelectionMode::PartPseudoElement) {
            if (scopeOrdinal <= ScopeOrdinal::ContainingHost)
                return hostForScopeOrdinal(element, scopeOrdinal);
            ASSERT(scopeOrdinal == ScopeOrdinal::Element);
            return element.shadowHost();
        }
        // ::slotted() selectors can query containers inside the shadow tree, including the slot itself.
        if (scopeOrdinal >= ScopeOrdinal::FirstSlot && scopeOrdinal <= ScopeOrdinal::SlotLimit)
            return assignedSlotForScopeOrdinal(element, scopeOrdinal);
        return nullptr;
    };

    if (RefPtr originatingElement = findOriginatingElement()) {
        // For selectors with pseudo elements, query containers can be established by the shadow-including inclusive ancestors of the ultimate originating element.
        for (RefPtr ancestor = originatingElement; ancestor; ancestor = ancestor->parentOrShadowHostElement()) {
            if (isContainerForQuery(*ancestor.get(), originatingElement.get()))
                return ancestor.get();
        }
        return nullptr;
    }

    if (selectionMode == SelectionMode::PseudoElement) {
        if (isContainerForQuery(element))
            return &element;
    }

    if (evaluationState && !requiredAxes.isEmpty()) {
        for (auto& container : makeReversedRange(evaluationState->sizeQueryContainers)) {
            if (isContainerForQuery(container))
                return container.ptr();
        }
        return { };
    }

    for (RefPtr ancestor = element.parentOrShadowHostElement(); ancestor; ancestor = ancestor->parentOrShadowHostElement()) {
        if (isContainerForQuery(*ancestor.get()))
            return ancestor.get();
    }
    return { };
}

}
