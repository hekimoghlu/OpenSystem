/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 14, 2024.
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
#include "RenderTextControlMultiLine.h"

#include "HTMLNames.h"
#include "HTMLTextAreaElement.h"
#include "HitTestResult.h"
#include "LocalFrame.h"
#include "RenderBoxInlines.h"
#include "RenderBoxModelObjectInlines.h"
#include "RenderLayerScrollableArea.h"
#include "RenderStyleSetters.h"
#include "ShadowRoot.h"
#include "StyleInheritedData.h"
#include "TextControlInnerElements.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(RenderTextControlMultiLine);

RenderTextControlMultiLine::RenderTextControlMultiLine(HTMLTextAreaElement& element, RenderStyle&& style)
    : RenderTextControl(Type::TextControlMultiLine, element, WTFMove(style))
{
    ASSERT(isRenderTextControlMultiLine());
}

// Do not add any code in below destructor. Add it to willBeDestroyed() instead.
RenderTextControlMultiLine::~RenderTextControlMultiLine() = default;

HTMLTextAreaElement& RenderTextControlMultiLine::textAreaElement() const
{
    return downcast<HTMLTextAreaElement>(RenderTextControl::textFormControlElement());
}

bool RenderTextControlMultiLine::nodeAtPoint(const HitTestRequest& request, HitTestResult& result, const HitTestLocation& locationInContainer, const LayoutPoint& accumulatedOffset, HitTestAction hitTestAction)
{
    if (!RenderTextControl::nodeAtPoint(request, result, locationInContainer, accumulatedOffset, hitTestAction))
        return false;

    const LayoutPoint adjustedPoint(accumulatedOffset + location());
    if (isPointInOverflowControl(result, locationInContainer.point(), adjustedPoint))
        return true;

    if (result.innerNode() == &textAreaElement() || result.innerNode() == innerTextElement())
        hitInnerTextElement(result, locationInContainer.point(), accumulatedOffset);

    return true;
}

float RenderTextControlMultiLine::getAverageCharWidth()
{
#if !PLATFORM(IOS_FAMILY)
    // Since Lucida Grande is the default font, we want this to match the width
    // of Courier New, the default font for textareas in IE, Firefox and Safari Win.
    // 1229 is the avgCharWidth value in the OS/2 table for Courier New.
    if (style().fontCascade().firstFamily() == "Lucida Grande"_s)
        return scaleEmToUnits(1229);
#endif

    return RenderTextControl::getAverageCharWidth();
}

LayoutUnit RenderTextControlMultiLine::preferredContentLogicalWidth(float charWidth) const
{
    float width = ceilf(charWidth * textAreaElement().cols());

    auto overflow = writingMode().isHorizontal() ? style().overflowY() : style().overflowX();

    // We are able to have a vertical scrollbar if the overflow style is scroll or auto
    if ((overflow == Overflow::Scroll) || (overflow == Overflow::Auto))
        width += scrollbarThickness();

    return LayoutUnit(width);
}

LayoutUnit RenderTextControlMultiLine::computeControlLogicalHeight(LayoutUnit lineHeight, LayoutUnit nonContentHeight) const
{
    return lineHeight * textAreaElement().rows() + nonContentHeight;
}

LayoutUnit RenderTextControlMultiLine::baselinePosition(FontBaseline baselineType, bool firstLine, LineDirectionMode direction, LinePositionMode linePositionMode) const
{
    return RenderBox::baselinePosition(baselineType, firstLine, direction, linePositionMode);
}

void RenderTextControlMultiLine::layoutExcludedChildren(bool relayoutChildren)
{
    RenderTextControl::layoutExcludedChildren(relayoutChildren);
    HTMLElement* placeholder = textFormControlElement().placeholderElement();
    RenderElement* placeholderRenderer = placeholder ? placeholder->renderer() : 0;
    if (!placeholderRenderer)
        return;
    if (CheckedPtr placeholderBox = dynamicDowncast<RenderBox>(placeholderRenderer)) {
        placeholderBox->mutableStyle().setLogicalWidth(Length(contentBoxLogicalWidth() - placeholderBox->borderAndPaddingLogicalWidth(), LengthType::Fixed));
        placeholderBox->layoutIfNeeded();
        placeholderBox->setX(borderLeft() + paddingLeft());
        placeholderBox->setY(borderTop() + paddingTop());
    }
}
    
}
