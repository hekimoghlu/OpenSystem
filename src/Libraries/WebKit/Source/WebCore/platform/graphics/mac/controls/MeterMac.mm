/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 13, 2023.
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
#import "config.h"
#import "MeterMac.h"

#if PLATFORM(MAC)

#import "ControlFactoryMac.h"
#import "GraphicsContext.h"
#import "LocalDefaultSystemAppearance.h"
#import "MeterPart.h"
#import <wtf/BlockObjCExceptions.h>
#import <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(MeterMac);

MeterMac::MeterMac(MeterPart& owningMeterPart, ControlFactoryMac& controlFactory, NSLevelIndicatorCell* levelIndicatorCell)
    : ControlMac(owningMeterPart, controlFactory)
    , m_levelIndicatorCell(levelIndicatorCell)
{
    ASSERT(m_levelIndicatorCell);
}

void MeterMac::updateCellStates(const FloatRect& rect, const ControlStyle& style)
{
    BEGIN_BLOCK_OBJC_EXCEPTIONS

    ControlMac::updateCellStates(rect, style);

    [m_levelIndicatorCell setUserInterfaceLayoutDirection:style.states.contains(ControlStyle::State::InlineFlippedWritingMode) ? NSUserInterfaceLayoutDirectionRightToLeft : NSUserInterfaceLayoutDirectionLeftToRight];

    auto& meterPart = owningMeterPart();
    
    // Because NSLevelIndicatorCell does not support optimum-in-the-middle type coloring,
    // we explicitly control the color instead giving low and high value to NSLevelIndicatorCell as is.
    switch (meterPart.gaugeRegion()) {
    case MeterPart::GaugeRegion::Optimum:
        // Make meter the green
        [m_levelIndicatorCell setWarningValue:meterPart.value() + 1];
        [m_levelIndicatorCell setCriticalValue:meterPart.value() + 2];
        break;
    case MeterPart::GaugeRegion::Suboptimal:
        // Make the meter yellow
        [m_levelIndicatorCell setWarningValue:meterPart.value() - 1];
        [m_levelIndicatorCell setCriticalValue:meterPart.value() + 1];
        break;
    case MeterPart::GaugeRegion::EvenLessGood:
        // Make the meter red
        [m_levelIndicatorCell setWarningValue:meterPart.value() - 2];
        [m_levelIndicatorCell setCriticalValue:meterPart.value() - 1];
        break;
    }

    [m_levelIndicatorCell setObjectValue:@(meterPart.value())];
    [m_levelIndicatorCell setMinValue:meterPart.minimum()];
    [m_levelIndicatorCell setMaxValue:meterPart.maximum()];

    END_BLOCK_OBJC_EXCEPTIONS
}

FloatSize MeterMac::sizeForBounds(const FloatRect& bounds, const ControlStyle& style) const
{
    auto isVerticalWritingMode = style.states.contains(ControlStyle::State::VerticalWritingMode);

    auto logicalSize = isVerticalWritingMode ? bounds.size().transposedSize() : bounds.size();

    // Makes enough room for cell's intrinsic size.
    NSSize cellSize = [m_levelIndicatorCell cellSizeForBounds:IntRect({ }, IntSize(logicalSize))];
    logicalSize = { std::max<float>(logicalSize.width(), cellSize.width), std::max<float>(logicalSize.height(), cellSize.height) };

    return isVerticalWritingMode ? logicalSize.transposedSize() : logicalSize;
}

void MeterMac::draw(GraphicsContext& context, const FloatRoundedRect& borderRect, float deviceScaleFactor, const ControlStyle& style)
{
    LocalDefaultSystemAppearance localAppearance(style.states.contains(ControlStyle::State::DarkAppearance), style.accentColor);

    GraphicsContextStateSaver stateSaver(context);

    auto rect = borderRect.rect();

    if (style.states.contains(ControlStyle::State::VerticalWritingMode)) {
        rect.setSize(rect.size().transposedSize());

        context.translate(rect.height(), 0);
        context.translate(rect.location());
        context.rotate(piOverTwoFloat);
        context.translate(-rect.location());
    }

    drawCell(context, rect, deviceScaleFactor, style, m_levelIndicatorCell.get());
}

} // namespace WebCore

#endif // PLATFORM(MAC)
