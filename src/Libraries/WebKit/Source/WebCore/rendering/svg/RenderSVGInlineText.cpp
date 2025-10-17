/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 14, 2024.
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
#include "RenderSVGInlineText.h"

#include "CSSFontSelector.h"
#include "FloatConversion.h"
#include "FloatQuad.h"
#include "InlineIteratorSVGTextBox.h"
#include "InlineRunAndOffset.h"
#include "LegacyRenderSVGRoot.h"
#include "RenderAncestorIterator.h"
#include "RenderBlock.h"
#include "RenderObjectInlines.h"
#include "RenderSVGText.h"
#include "SVGElementTypeHelpers.h"
#include "SVGInlineTextBoxInlines.h"
#include "SVGLayerTransformComputation.h"
#include "SVGRenderingContext.h"
#include "SVGRootInlineBox.h"
#include "SVGTextBoxPainter.h"
#include "StyleFontSizeFunctions.h"
#include "StyleResolver.h"
#include "VisiblePosition.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(RenderSVGInlineText);

static String applySVGWhitespaceRules(const String& string, bool preserveWhiteSpace)
{
    String newString = string;
    if (preserveWhiteSpace) {
        // Spec: When xml:space="preserve", the SVG user agent will do the following using a
        // copy of the original character data content. It will convert all newline and tab
        // characters into space characters. Then, it will draw all space characters, including
        // leading, trailing and multiple contiguous space characters.
        newString = makeStringByReplacingAll(newString, '\t', ' ');
        newString = makeStringByReplacingAll(newString, '\n', ' ');
        newString = makeStringByReplacingAll(newString, '\r', ' ');
        return newString;
    }

    // Spec: When xml:space="default", the SVG user agent will do the following using a
    // copy of the original character data content. First, it will remove all newline
    // characters. Then it will convert all tab characters into space characters.
    // Then, it will strip off all leading and trailing space characters.
    // Then, all contiguous space characters will be consolidated.
    newString = makeStringByReplacingAll(newString, '\n', ""_s);
    newString = makeStringByReplacingAll(newString, '\r', ""_s);
    newString = makeStringByReplacingAll(newString, '\t', ' ');
    return newString;
}

RenderSVGInlineText::RenderSVGInlineText(Text& textNode, const String& string)
    : RenderText(Type::SVGInlineText, textNode, applySVGWhitespaceRules(string, false))
    , m_scalingFactor(1)
    , m_layoutAttributes(*this)
{
    ASSERT(isRenderSVGInlineText());
}

RenderSVGInlineText::~RenderSVGInlineText() = default;

String RenderSVGInlineText::originalText() const
{
    return textNode().data();
}

void RenderSVGInlineText::setRenderedText(const String& text)
{
    RenderText::setRenderedText(text);
    if (auto* textAncestor = RenderSVGText::locateRenderSVGTextAncestor(*this))
        textAncestor->subtreeTextDidChange(this);
}

void RenderSVGInlineText::styleDidChange(StyleDifference diff, const RenderStyle* oldStyle)
{
    RenderText::styleDidChange(diff, oldStyle);
    updateScaledFont();

    bool newPreserves = style().whiteSpaceCollapse() == WhiteSpaceCollapse::Preserve;
    bool oldPreserves = oldStyle ? oldStyle->whiteSpaceCollapse() == WhiteSpaceCollapse::Preserve : false;
    if (oldPreserves && !newPreserves) {
        setText(applySVGWhitespaceRules(originalText(), false), true);
        return;
    }

    if (!oldPreserves && newPreserves) {
        setText(applySVGWhitespaceRules(originalText(), true), true);
        return;
    }

    if (diff != StyleDifference::Layout)
        return;

    // The text metrics may be influenced by style changes.
    if (auto* textAncestor = RenderSVGText::locateRenderSVGTextAncestor(*this))
        textAncestor->setNeedsLayout();
}

std::unique_ptr<LegacyInlineTextBox> RenderSVGInlineText::createTextBox()
{
    auto box = makeUnique<SVGInlineTextBox>(*this);
    box->setHasVirtualLogicalHeight();
    return box; 
}

FloatRect RenderSVGInlineText::floatLinesBoundingBox() const
{
    FloatRect boundingBox;
    for (auto& box : InlineIterator::svgTextBoxesFor(*this))
        boundingBox.unite(box.calculateBoundariesIncludingSVGTransform());

    return boundingBox;
}

IntRect RenderSVGInlineText::linesBoundingBox() const
{
    return enclosingIntRect(floatLinesBoundingBox());
}

bool RenderSVGInlineText::characterStartsNewTextChunk(int position) const
{
    ASSERT(position >= 0);
    ASSERT(position < static_cast<int>(text().length()));

    // Each <textPath> element starts a new text chunk, regardless of any x/y values.
    if (!position && parent()->isRenderSVGTextPath() && !previousSibling())
        return true;

    const SVGCharacterDataMap::const_iterator it = m_layoutAttributes.characterDataMap().find(static_cast<unsigned>(position + 1));
    if (it == m_layoutAttributes.characterDataMap().end())
        return false;

    return !SVGTextLayoutAttributes::isEmptyValue(it->value.x) || !SVGTextLayoutAttributes::isEmptyValue(it->value.y);
}

static int offsetForPositionInFragment(const InlineIterator::SVGTextBox& textBox, const SVGTextFragment& fragment, float position)
{
    float scalingFactor = textBox.renderer().scalingFactor();
    ASSERT(scalingFactor);

    TextRun textRun = constructTextRun(textBox.renderer().text(), textBox.direction(), textBox.style(), fragment);

    // Eventually handle lengthAdjust="spacingAndGlyphs".
    // FIXME: Handle vertical text.
    AffineTransform fragmentTransform;
    fragment.buildFragmentTransform(fragmentTransform);
    if (!fragmentTransform.isIdentity())
        textRun.setHorizontalGlyphStretch(narrowPrecisionToFloat(fragmentTransform.xScale()));

    const bool includePartialGlyphs = true;
    return fragment.characterOffset - textBox.start() + textBox.renderer().scaledFont().offsetForPosition(textRun, position * scalingFactor, includePartialGlyphs);
}

VisiblePosition RenderSVGInlineText::positionForPoint(const LayoutPoint& point, HitTestSource, const RenderFragmentContainer*)
{
    if (!InlineIterator::lineLeftmostTextBoxFor(*this) || text().isEmpty())
        return createVisiblePosition(0, Affinity::Downstream);

    float baseline = m_scaledFont.metricsOfPrimaryFont().ascent();

    RenderBlock* containingBlock = this->containingBlock();
    ASSERT(containingBlock);

    // Map local point to absolute point, as the character origins stored in the text fragments use absolute coordinates.
    FloatPoint absolutePoint(point);
    absolutePoint.moveBy(containingBlock->location());

    float closestDistance = std::numeric_limits<float>::max();
    float closestDistancePosition = 0;
    const SVGTextFragment* closestDistanceFragment = nullptr;
    InlineIterator::SVGTextBoxIterator closestDistanceBox;

    AffineTransform fragmentTransform;
    for (auto& box : InlineIterator::svgTextBoxesFor(*this)) {
        auto& fragments = box.textFragments();

        unsigned textFragmentsSize = fragments.size();
        for (unsigned i = 0; i < textFragmentsSize; ++i) {
            const SVGTextFragment& fragment = fragments.at(i);
            FloatRect fragmentRect(fragment.x, fragment.y - baseline, fragment.width, fragment.height);
            fragment.buildFragmentTransform(fragmentTransform);
            if (!fragmentTransform.isIdentity())
                fragmentRect = fragmentTransform.mapRect(fragmentRect);

            float distance = powf(fragmentRect.x() - absolutePoint.x(), 2) +
                             powf(fragmentRect.y() + fragmentRect.height() / 2 - absolutePoint.y(), 2);

            if (distance < closestDistance) {
                closestDistance = distance;
                closestDistanceBox = box;
                closestDistanceFragment = &fragment;
                closestDistancePosition = fragmentRect.x();
            }
        }
    }

    if (!closestDistanceFragment)
        return createVisiblePosition(0, Affinity::Downstream);

    int offset = offsetForPositionInFragment(*closestDistanceBox, *closestDistanceFragment, absolutePoint.x() - closestDistancePosition);
    return createVisiblePosition(offset + closestDistanceBox->start(), offset > 0 ? Affinity::Upstream : Affinity::Downstream);
}

void RenderSVGInlineText::updateScaledFont()
{
    if (computeNewScaledFontForStyle(*this, style(), m_scalingFactor, m_scaledFont))
        m_canUseSimplifiedTextMeasuring = { };
}

float RenderSVGInlineText::computeScalingFactorForRenderer(const RenderObject& renderer)
{
    if (renderer.document().settings().layerBasedSVGEngineEnabled()) {
        if (const auto* layerRenderer = lineageOfType<RenderLayerModelObject>(renderer).first())
            return SVGLayerTransformComputation(*layerRenderer).calculateScreenFontSizeScalingFactor();
    }
    return SVGRenderingContext::calculateScreenFontSizeScalingFactor(renderer);
}

bool RenderSVGInlineText::computeNewScaledFontForStyle(const RenderObject& renderer, const RenderStyle& style, float& scalingFactor, FontCascade& scaledFont)
{
    // Alter font-size to the right on-screen value to avoid scaling the glyphs themselves, except when GeometricPrecision is specified
    scalingFactor = computeScalingFactorForRenderer(renderer);
    if (!scalingFactor) {
        scalingFactor = 1;
        scaledFont = style.fontCascade();
        return false;
    }

    if (style.fontDescription().textRenderingMode() == TextRenderingMode::GeometricPrecision)
        scalingFactor = 1;

    auto fontDescription = style.fontDescription();

    // FIXME: We need to better handle the case when we compute very small fonts below (below 1pt).
    fontDescription.setComputedSize(Style::computedFontSizeFromSpecifiedSizeForSVGInlineText(fontDescription.specifiedSize(), fontDescription.isAbsoluteSize(), scalingFactor, renderer.protectedDocument()));

    // SVG controls its own glyph orientation, so don't allow writing-mode
    // to affect it.
    if (fontDescription.orientation() != FontOrientation::Horizontal)
        fontDescription.setOrientation(FontOrientation::Horizontal);

    scaledFont = FontCascade(WTFMove(fontDescription));
    scaledFont.update(renderer.document().protectedFontSelector().ptr());
    return true;
}

}
