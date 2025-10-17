/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 28, 2022.
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
#include "RenderMeter.h"

#include "HTMLMeterElement.h"
#include "HTMLNames.h"
#include "RenderBoxInlines.h"
#include "RenderBoxModelObjectInlines.h"
#include "RenderTheme.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

using namespace HTMLNames;

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(RenderMeter);

RenderMeter::RenderMeter(HTMLElement& element, RenderStyle&& style)
    : RenderBlockFlow(Type::Meter, element, WTFMove(style))
{
    ASSERT(isRenderMeter());
}

RenderMeter::~RenderMeter() = default;

HTMLMeterElement* RenderMeter::meterElement() const
{
    ASSERT(element());

    if (auto* meterElement = dynamicDowncast<HTMLMeterElement>(*element()))
        return meterElement;

    ASSERT(element()->shadowHost());
    return downcast<HTMLMeterElement>(element()->shadowHost());
}

void RenderMeter::updateLogicalWidth()
{
    RenderBox::updateLogicalWidth();

    auto frameSize = theme().meterSizeForBounds(*this, snappedIntRect(frameRect()));
    setLogicalWidth(LayoutUnit(isHorizontalWritingMode() ? frameSize.width() : frameSize.height()));
}

RenderBox::LogicalExtentComputedValues RenderMeter::computeLogicalHeight(LayoutUnit logicalHeight, LayoutUnit logicalTop) const
{
    auto computedValues = RenderBox::computeLogicalHeight(logicalHeight, logicalTop);
    LayoutRect frame = frameRect();
    if (isHorizontalWritingMode())
        frame.setHeight(computedValues.m_extent);
    else
        frame.setWidth(computedValues.m_extent);
    auto frameSize = theme().meterSizeForBounds(*this, snappedIntRect(frame));
    computedValues.m_extent = isHorizontalWritingMode() ? frameSize.height() : frameSize.width();
    return computedValues;
}

void RenderMeter::updateFromElement()
{
    repaint();
}

} // namespace WebCore
