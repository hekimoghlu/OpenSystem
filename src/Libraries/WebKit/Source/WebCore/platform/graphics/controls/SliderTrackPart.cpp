/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 5, 2024.
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
#include "SliderTrackPart.h"

#include "ControlFactory.h"
#include "GraphicsContext.h"

namespace WebCore {

Ref<SliderTrackPart> SliderTrackPart::create(StyleAppearance type)
{
    return adoptRef(*new SliderTrackPart(type, { }, { }, { }, 0));
}

Ref<SliderTrackPart> SliderTrackPart::create(StyleAppearance type, const IntSize& thumbSize, const IntRect& trackBounds, Vector<double>&& tickRatios, double thumbPosition)
{
    return adoptRef(*new SliderTrackPart(type, thumbSize, trackBounds, WTFMove(tickRatios), thumbPosition));
}

SliderTrackPart::SliderTrackPart(StyleAppearance type, const IntSize& thumbSize, const IntRect& trackBounds, Vector<double>&& tickRatios, double thumbPosition)
    : ControlPart(type)
    , m_thumbSize(thumbSize)
    , m_trackBounds(trackBounds)
    , m_tickRatios(WTFMove(tickRatios))
    , m_thumbPosition(thumbPosition)
{
    ASSERT(type == StyleAppearance::SliderHorizontal || type == StyleAppearance::SliderVertical);
}

void SliderTrackPart::drawTicks(GraphicsContext& context, const FloatRect& rect, const ControlStyle& style) const
{
    if (m_tickRatios.isEmpty())
        return;

    static constexpr FloatSize sliderTickSize = { 1, 3 };
    static constexpr int sliderTickOffsetFromTrackCenter = -9;

    bool isHorizontal = type() == StyleAppearance::SliderHorizontal;

    auto trackBounds = m_trackBounds;
    trackBounds.moveBy(IntPoint(rect.location()));

    auto tickSize = isHorizontal? sliderTickSize : sliderTickSize.transposedSize();
    tickSize.scale(style.zoomFactor);

    FloatPoint tickLocation;
    float tickRegionMargin = 0;
    float tickRegionWidth = 0;
    float offsetFromTrackCenter = sliderTickOffsetFromTrackCenter * style.zoomFactor;

    if (isHorizontal) {
        tickLocation = { 0, rect.center().y() + offsetFromTrackCenter };
        tickRegionMargin = trackBounds.x() + (m_thumbSize.width() - tickSize.width()) / 2.0;
        tickRegionWidth = trackBounds.width() - m_thumbSize.width();
    } else {
        tickLocation = { rect.center().x() + offsetFromTrackCenter, 0 };
        tickRegionMargin = trackBounds.y() + (m_thumbSize.height() - tickSize.height()) / 2.0;
        tickRegionWidth = trackBounds.height() - m_thumbSize.height();
    }

    auto tickRect = FloatRect { tickLocation, tickSize };

    GraphicsContextStateSaver stateSaver(context);
    context.setFillColor(style.textColor);

    bool isVerticalWritingMode = style.states.contains(ControlStyle::State::VerticalWritingMode);
    bool isInlineFlippedWritingMode = style.states.contains(ControlStyle::State::InlineFlippedWritingMode);
    bool isInlineFlipped = (!isHorizontal && !isVerticalWritingMode) || isInlineFlippedWritingMode;

    for (auto tickRatio : m_tickRatios) {
        double value = isInlineFlipped ? 1.0 - tickRatio : tickRatio;
        double tickPosition = round(tickRegionMargin + tickRegionWidth * value);
        if (isHorizontal)
            tickRect.setX(tickPosition);
        else
            tickRect.setY(tickPosition);
        context.fillRect(tickRect);
    }
}

std::unique_ptr<PlatformControl> SliderTrackPart::createPlatformControl()
{
    return controlFactory().createPlatformSliderTrack(*this);
}

} // namespace WebCore
