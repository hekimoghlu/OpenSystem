/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 28, 2022.
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
#include "MathMLStyle.h"

#if ENABLE(MATHML)

#include "MathMLElement.h"
#include "MathMLNames.h"
#include "RenderMathMLBlock.h"
#include "RenderMathMLFraction.h"
#include "RenderMathMLMath.h"
#include "RenderMathMLRoot.h"
#include "RenderMathMLScripts.h"
#include "RenderMathMLToken.h"
#include "RenderMathMLUnderOver.h"

namespace WebCore {

using namespace MathMLNames;

Ref<MathMLStyle> MathMLStyle::create()
{
    return adoptRef(*new MathMLStyle());
}

const MathMLStyle* MathMLStyle::getMathMLStyle(RenderObject* renderer)
{
    // FIXME: Should we make RenderMathMLTable derive from RenderMathMLBlock in order to simplify this?
    if (auto* mathMLTable = dynamicDowncast<RenderMathMLTable>(renderer))
        return &mathMLTable->mathMLStyle();
    if (auto* mathMLBlock = dynamicDowncast<RenderMathMLBlock>(renderer))
        return &mathMLBlock->mathMLStyle();
    return nullptr;
}

void MathMLStyle::resolveMathMLStyleTree(RenderObject* renderer)
{
    for (auto* child = renderer; child; child = child->nextInPreOrder(renderer)) {
        // FIXME: Should we make RenderMathMLTable derive from RenderMathMLBlock in order to simplify this?
        if (auto* mathMLTable = dynamicDowncast<RenderMathMLTable>(child))
            mathMLTable->mathMLStyle().resolveMathMLStyle(child);
        else if (auto* mathMLBlock = dynamicDowncast<RenderMathMLBlock>(child))
            mathMLBlock->mathMLStyle().resolveMathMLStyle(child);
    }
}

RenderObject* MathMLStyle::getMathMLParentNode(RenderObject* renderer)
{
    auto* parentRenderer = renderer->parent();

    while (parentRenderer && !(is<RenderMathMLTable>(parentRenderer) || is<RenderMathMLBlock>(parentRenderer)))
        parentRenderer = parentRenderer->parent();

    return parentRenderer;
}

void MathMLStyle::updateStyleIfNeeded(RenderObject* renderer, MathMLElement::MathVariant oldMathVariant)
{
    // RenderMathMLFencedOperator does not support mathvariant transforms.
    // See https://bugs.webkit.org/show_bug.cgi?id=160509#c1.
    if (oldMathVariant != m_mathVariant) {
        auto* mathMLToken = dynamicDowncast<RenderMathMLToken>(renderer);
        if (mathMLToken && !mathMLToken->isAnonymous())
            mathMLToken->updateTokenContent();
    }
}

void MathMLStyle::resolveMathMLStyle(RenderObject* renderer)
{
    ASSERT(renderer);

    MathMLElement::MathVariant oldMathVariant = m_mathVariant;
    auto* parentRenderer = getMathMLParentNode(renderer);
    const MathMLStyle* parentStyle = getMathMLStyle(parentRenderer);

    // By default, we just inherit the style from our parent.
    m_mathVariant = MathMLElement::MathVariant::None;
    if (parentStyle) {
        setMathVariant(parentStyle->mathVariant());
    }

    // Early return for anonymous renderers.
    if (renderer->isAnonymous()) {
        updateStyleIfNeeded(renderer, oldMathVariant);
        return;
    }

    // The mathvariant attributes override the default behavior.
    if (auto* element = dynamicDowncast<MathMLElement>(downcast<RenderElement>(renderer)->element())) {
        std::optional<MathMLElement::MathVariant> mathVariant = element->specifiedMathVariant();
        if (mathVariant)
            m_mathVariant = mathVariant.value();
    }
    updateStyleIfNeeded(renderer, oldMathVariant);
}

}

#endif // ENABLE(MATHML)
