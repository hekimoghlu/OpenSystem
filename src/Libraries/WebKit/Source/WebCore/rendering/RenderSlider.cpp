/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 28, 2022.
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
#include "RenderSlider.h"

#include "CSSPropertyNames.h"
#include "Document.h"
#include "Event.h"
#include "HTMLInputElement.h"
#include "HTMLNames.h"
#include "HTMLParserIdioms.h"
#include "LocalFrame.h"
#include "MouseEvent.h"
#include "Node.h"
#include "RenderBoxInlines.h"
#include "RenderBoxModelObjectInlines.h"
#include "RenderElementInlines.h"
#include "RenderLayer.h"
#include "RenderTheme.h"
#include "RenderView.h"
#include "ShadowRoot.h"
#include "SliderThumbElement.h"
#include "StepRange.h"
#include "StyleResolver.h"
#include <wtf/MathExtras.h>
#include <wtf/StackStats.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(RenderSlider);

const int RenderSlider::defaultTrackLength = 129;

RenderSlider::RenderSlider(HTMLInputElement& element, RenderStyle&& style)
    : RenderFlexibleBox(Type::Slider, element, WTFMove(style))
{
    // We assume RenderSlider works only with <input type=range>.
    ASSERT(element.isRangeControl());
    ASSERT(isRenderSlider());
}

RenderSlider::~RenderSlider() = default;

HTMLInputElement& RenderSlider::element() const
{
    return downcast<HTMLInputElement>(nodeForNonAnonymous());
}

LayoutUnit RenderSlider::baselinePosition(FontBaseline, bool /*firstLine*/, LineDirectionMode, LinePositionMode) const
{
    // FIXME: Patch this function for writing-mode.
    return height() + marginTop();
}

void RenderSlider::computeIntrinsicLogicalWidths(LayoutUnit& minLogicalWidth, LayoutUnit& maxLogicalWidth) const
{
    if (shouldApplySizeOrInlineSizeContainment()) {
        if (auto width = explicitIntrinsicInnerLogicalWidth()) {
            minLogicalWidth = width.value();
            maxLogicalWidth = width.value();
        }
        return;
    }
    maxLogicalWidth = defaultTrackLength * style().usedZoom();
    auto& logicalWidth = style().logicalWidth();
    if (logicalWidth.isCalculated())
        minLogicalWidth = std::max(0_lu, valueForLength(logicalWidth, 0_lu));
    else if (!logicalWidth.isPercent())
        minLogicalWidth = maxLogicalWidth;
}

void RenderSlider::computePreferredLogicalWidths()
{
    m_minPreferredLogicalWidth = 0;
    m_maxPreferredLogicalWidth = 0;

    if (style().logicalWidth().isFixed())
        m_minPreferredLogicalWidth = m_maxPreferredLogicalWidth = adjustContentBoxLogicalWidthForBoxSizing(style().logicalWidth());
    else
        computeIntrinsicLogicalWidths(m_minPreferredLogicalWidth, m_maxPreferredLogicalWidth);

    RenderBox::computePreferredLogicalWidths(style().logicalMinWidth(), style().logicalMaxWidth(), writingMode().isHorizontal() ? horizontalBorderAndPaddingExtent() : verticalBorderAndPaddingExtent());

    setPreferredLogicalWidthsDirty(false); 
}

bool RenderSlider::inDragMode() const
{
    return element().sliderThumbElement()->active();
}

double RenderSlider::valueRatio() const
{
    auto& element = this->element();

    auto min = element.minimum();
    auto max = element.maximum();
    auto value = element.valueAsNumber();

    if (max <= min)
        return 0;
    return (value - min) / (max - min);
}

} // namespace WebCore
