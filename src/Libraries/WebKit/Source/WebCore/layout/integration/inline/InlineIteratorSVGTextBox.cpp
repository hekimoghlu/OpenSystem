/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 16, 2023.
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
#include "InlineIteratorSVGTextBox.h"

#include "LayoutIntegrationLineLayout.h"
#include "RenderSVGText.h"
#include "SVGInlineTextBox.h"
#include "SVGRootInlineBox.h"
#include "SVGTextBoxPainter.h"
#include "SVGTextFragment.h"

namespace WebCore {
namespace InlineIterator {

SVGTextBox::SVGTextBox(PathVariant&& path)
    : TextBox(WTFMove(path))
{
}

FloatRect SVGTextBox::calculateBoundariesIncludingSVGTransform() const
{
    FloatRect textRect;

    float scalingFactor = renderer().scalingFactor();
    ASSERT(scalingFactor);

    float baseline = renderer().scaledFont().metricsOfPrimaryFont().ascent() / scalingFactor;
    for (auto& fragment : textFragments()) {
        auto fragmentRect = FloatRect { fragment.x, fragment.y - baseline, fragment.width, fragment.height };

        AffineTransform fragmentTransform;
        fragment.buildFragmentTransform(fragmentTransform);
        if (!fragmentTransform.isIdentity())
            fragmentRect = fragmentTransform.mapRect(fragmentRect);

        textRect.unite(fragmentRect);
    }
    return textRect;
}

LayoutRect SVGTextBox::localSelectionRect(unsigned start, unsigned end) const
{
    auto [clampedStart, clampedEnd] = selectableRange().clamp(start, end);

    if (clampedStart >= clampedEnd)
        return LayoutRect();

    auto& style = renderer().style();

    AffineTransform fragmentTransform;
    FloatRect selectionRect;
    unsigned fragmentStartPosition = 0;
    unsigned fragmentEndPosition = 0;

    for (auto& fragment : textFragments()) {
        fragmentStartPosition = clampedStart;
        fragmentEndPosition = clampedEnd;
        if (!mapStartEndPositionsIntoFragmentCoordinates(this->start(), fragment, fragmentStartPosition, fragmentEndPosition))
            continue;

        FloatRect fragmentRect = selectionRectForTextFragment(renderer(), direction(), fragment, fragmentStartPosition, fragmentEndPosition, style);
        fragment.buildFragmentTransform(fragmentTransform);
        if (!fragmentTransform.isIdentity())
            fragmentRect = fragmentTransform.mapRect(fragmentRect);

        selectionRect.unite(fragmentRect);
    }

    return enclosingIntRect(selectionRect);
}

const Vector<SVGTextFragment>& SVGTextBox::textFragments() const
{
    return WTF::switchOn(m_pathVariant, [&](auto& path) -> const Vector<SVGTextFragment>& {
        return path.svgTextFragments();
    });
}

const SVGInlineTextBox* SVGTextBox::legacyInlineBox() const
{
    return downcast<SVGInlineTextBox>(TextBox::legacyInlineBox());
}

SVGTextBoxIterator::SVGTextBoxIterator(Box::PathVariant&& path)
    : TextBoxIterator(WTFMove(path))
{
}

SVGTextBoxIterator::SVGTextBoxIterator(const Box& box)
    : TextBoxIterator(box)
{
}

SVGTextBoxIterator firstSVGTextBoxFor(const RenderSVGInlineText& text)
{
    if (auto* lineLayout = LayoutIntegration::LineLayout::containing(text)) {
        auto box = lineLayout->textBoxesFor(text);
        if (!box)
            return { };
        return { *box };
    }

    return { BoxLegacyPath { text.firstLegacyTextBox() } };
}

BoxRange<SVGTextBoxIterator> svgTextBoxesFor(const RenderSVGInlineText& text)
{
    return { firstSVGTextBoxFor(text) };
}

SVGTextBoxIterator svgTextBoxFor(const SVGInlineTextBox* box)
{
    return { BoxLegacyPath { box } };
}

SVGTextBoxIterator svgTextBoxFor(const LayoutIntegration::InlineContent& inlineContent, size_t boxIndex)
{
    auto& box = inlineContent.displayContent().boxes[boxIndex];
    if (!box.isText() || !box.layoutBox().rendererForIntegration()->isRenderSVGInlineText())
        return { };
    return { BoxModernPath { inlineContent, boxIndex } };
}

SVGTextBox::Key makeKey(const SVGTextBox& textBox)
{
    return { &textBox.renderer(), textBox.start() };
}

BoxRange<BoxIterator> boxesFor(const RenderSVGText& svgText)
{
    if (auto* lineLayout = svgText.inlineLayout())
        return { BoxIterator { *lineLayout->firstRootInlineBox() } };

    return { BoxIterator { BoxLegacyPath { svgText.legacyRootBox() } } };
}

}
}
