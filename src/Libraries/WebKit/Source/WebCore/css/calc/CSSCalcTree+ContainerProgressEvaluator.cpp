/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 30, 2024.
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
#include "CSSCalcTree+ContainerProgressEvaluator.h"

#include "CSSCalcTree.h"
#include "ContainerQueryEvaluator.h"
#include "ContainerQueryFeatures.h"
#include "RenderBox.h"

namespace WebCore {
namespace CSSCalc {

std::optional<double> evaluateContainerProgress(const ContainerProgress& root, const Element& initialElement, const CSSToLengthConversionData& conversionData)
{
    // FIXME: This lookup loop is the same as the one used in CSSPrimitiveValue for resolving container units. Would be good to figure out a nice place to share this.

    RefPtr element = &initialElement;

    auto mode = conversionData.style()->pseudoElementType() == PseudoId::None
        ? Style::ContainerQueryEvaluator::SelectionMode::Element
        : Style::ContainerQueryEvaluator::SelectionMode::PseudoElement;

    while ((element = Style::ContainerQueryEvaluator::selectContainer({ }, root.container, *element, mode))) {
        auto* containerRenderer = dynamicDowncast<RenderBox>(element->renderer());
        if (containerRenderer && containerRenderer->hasEligibleContainmentForSizeQuery())
            return root.feature->valueInCanonicalUnits(*containerRenderer);

        // For pseudo-elements the element itself can be the container. Avoid looping forever.
        mode = Style::ContainerQueryEvaluator::SelectionMode::Element;
    }

    // "If no appropriate containers are found, container-progress() resolves its <size-feature> query against the small viewport size."
    auto view = conversionData.renderView();
    if (!view)
        return { };

    return root.feature->valueInCanonicalUnits(*view, *conversionData.style());
}

} // namespace CSSCalc
} // namespace WebCore
