/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 8, 2022.
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
#include "RenderTreeBuilderMathML.h"

#if ENABLE(MATHML)

#include "RenderMathMLFenced.h"
#include "RenderMathMLFencedOperator.h"
#include "RenderStyleSetters.h"
#include "RenderTreeBuilderBlock.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RenderTreeBuilder::MathML);

RenderTreeBuilder::MathML::MathML(RenderTreeBuilder& builder)
    : m_builder(builder)
{
}

RenderPtr<RenderMathMLFencedOperator> RenderTreeBuilder::MathML::createMathMLOperator(RenderMathMLFenced& parent, const String& operatorString,
    MathMLOperatorDictionary::Form form, MathMLOperatorDictionary::Flag flag)
{
    RenderPtr<RenderMathMLFencedOperator> newOperator = createRenderer<RenderMathMLFencedOperator>(parent.document(), RenderStyle::createAnonymousStyleWithDisplay(parent.style(), DisplayType::Block), operatorString, form, flag);
    newOperator->initializeStyle();
    return newOperator;
}

void RenderTreeBuilder::MathML::makeFences(RenderMathMLFenced& parent)
{
    auto openFence = createMathMLOperator(parent, parent.openingBrace(), MathMLOperatorDictionary::Prefix, MathMLOperatorDictionary::Fence);
    m_builder.blockBuilder().attach(parent, WTFMove(openFence), parent.firstChild());

    auto closeFence = createMathMLOperator(parent, parent.closingBrace(), MathMLOperatorDictionary::Postfix, MathMLOperatorDictionary::Fence);
    parent.setCloseFenceRenderer(*closeFence);
    m_builder.blockBuilder().attach(parent, WTFMove(closeFence), nullptr);
}

void RenderTreeBuilder::MathML::attach(RenderMathMLFenced& parent, RenderPtr<RenderObject> child, RenderObject* beforeChild)
{
    // make the fences if the render object is empty
    if (!parent.firstChild()) {
        parent.updateFromElement();
        makeFences(parent);
    }
    // FIXME: Adding or removing a child should possibly cause all later separators to shift places
    // if they're different, as later child positions change by +1 or -1.
    // This should also handle surrogate pairs. See https://bugs.webkit.org/show_bug.cgi?id=125938.
    RenderPtr<RenderMathMLFencedOperator> separatorRenderer;
    auto* separators = parent.separators();
    if (separators) {
        unsigned count = 0;
        for (Node* position = child->node(); position; position = position->previousSibling()) {
            if (position->isElementNode())
                count++;
        }
        if (!beforeChild) {
            // We're adding at the end (before the closing fence), so a new separator would go before the new child, not after it.
            --count;
        }
        // |count| is now the number of element children that will be before our new separator, i.e. it's the 1-based index of the separator.

        if (count > 0) {
            UChar separator;

            // Use the last separator if we've run out of specified separators.
            if (count > separators->length())
                separator = (*separators)[separators->length() - 1];
            else
                separator = (*separators)[count - 1];

            StringBuilder stringBuilder;
            stringBuilder.append(separator);
            separatorRenderer = createMathMLOperator(parent, stringBuilder.toString(), MathMLOperatorDictionary::Infix, MathMLOperatorDictionary::Separator);
        }
    }

    if (beforeChild) {
        // Adding |x| before an existing |y| e.g. in element (y) - first insert our new child |x|, then its separator, to get (x, y).
        m_builder.blockBuilder().attach(parent, WTFMove(child), beforeChild);
        if (separatorRenderer)
            m_builder.blockBuilder().attach(parent, WTFMove(separatorRenderer), beforeChild);
    } else {
        // Adding |y| at the end of an existing element e.g. (x) - insert the separator first before the closing fence, then |y|, to get (x, y).
        if (separatorRenderer)
            m_builder.blockBuilder().attach(parent, WTFMove(separatorRenderer), parent.closeFenceRenderer());
        m_builder.blockBuilder().attach(parent, WTFMove(child), parent.closeFenceRenderer());
    }
}

}

#endif // ENABLE(MATHML)
