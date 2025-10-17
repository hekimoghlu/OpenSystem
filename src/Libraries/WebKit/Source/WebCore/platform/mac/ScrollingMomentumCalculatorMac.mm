/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 27, 2022.
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
#import "ScrollingMomentumCalculatorMac.h"

#if PLATFORM(MAC)

#import <pal/spi/mac/NSScrollingMomentumCalculatorSPI.h>

namespace WebCore {

static bool gEnablePlatformMomentumScrollingPrediction = true;

std::unique_ptr<ScrollingMomentumCalculator> ScrollingMomentumCalculator::create(const ScrollExtents& scrollExtents, const FloatPoint& initialOffset, const FloatSize& initialDelta, const FloatSize& initialVelocity)
{
    return makeUnique<ScrollingMomentumCalculatorMac>(scrollExtents, initialOffset, initialDelta, initialVelocity);
}

void ScrollingMomentumCalculator::setPlatformMomentumScrollingPredictionEnabled(bool enabled)
{
    gEnablePlatformMomentumScrollingPrediction = enabled;
}

ScrollingMomentumCalculatorMac::ScrollingMomentumCalculatorMac(const ScrollExtents& scrollExtents, const FloatPoint& initialOffset, const FloatSize& initialDelta, const FloatSize& initialVelocity)
    : ScrollingMomentumCalculator(scrollExtents, initialOffset, initialDelta, initialVelocity)
{
    m_initialDestinationOffset = predictedDestinationOffset();
    // We could compute m_requiresMomentumScrolling here, based on whether initialDelta is non-zero or we are in a rubber-banded state.
}

FloatPoint ScrollingMomentumCalculatorMac::scrollOffsetAfterElapsedTime(Seconds elapsedTime)
{
    if (!requiresMomentumScrolling())
        return destinationScrollOffset();

    return [ensurePlatformMomentumCalculator() positionAfterDuration:elapsedTime.value()];
}

FloatPoint ScrollingMomentumCalculatorMac::predictedDestinationOffset()
{
    ensurePlatformMomentumCalculator();

    if (!gEnablePlatformMomentumScrollingPrediction) {
        auto nonPlatformPredictedOffset = ScrollingMomentumCalculator::predictedDestinationOffset();
        // We need to make sure the _NSScrollingMomentumCalculator has the same idea of what offset we're shooting for.
        if (nonPlatformPredictedOffset != m_initialDestinationOffset)
            setMomentumCalculatorDestinationOffset(nonPlatformPredictedOffset);

        return nonPlatformPredictedOffset;
    }

    return m_initialDestinationOffset;
}

void ScrollingMomentumCalculatorMac::destinationScrollOffsetDidChange()
{
    setMomentumCalculatorDestinationOffset(destinationScrollOffset());
}

void ScrollingMomentumCalculatorMac::setMomentumCalculatorDestinationOffset(FloatPoint scrollOffset)
{
    _NSScrollingMomentumCalculator *calculator = ensurePlatformMomentumCalculator();
    calculator.destinationOrigin = scrollOffset;
    [calculator calculateToReachDestination];
}

Seconds ScrollingMomentumCalculatorMac::animationDuration()
{
    if (!requiresMomentumScrolling())
        return 0_s;

    return Seconds([ensurePlatformMomentumCalculator() durationUntilStop]);
}

bool ScrollingMomentumCalculatorMac::requiresMomentumScrolling()
{
    if (m_requiresMomentumScrolling == std::nullopt)
        m_requiresMomentumScrolling = m_initialScrollOffset != destinationScrollOffset() || m_initialVelocity.area();
    return m_requiresMomentumScrolling.value();
}

_NSScrollingMomentumCalculator *ScrollingMomentumCalculatorMac::ensurePlatformMomentumCalculator()
{
    if (m_platformMomentumCalculator)
        return m_platformMomentumCalculator.get();

    NSPoint origin = m_initialScrollOffset;
    NSRect contentFrame = NSMakeRect(0, 0, m_scrollExtents.contentsSize.width(), m_scrollExtents.contentsSize.height());
    NSPoint velocity = NSMakePoint(m_initialVelocity.width(), m_initialVelocity.height());
    m_platformMomentumCalculator = adoptNS([[_NSScrollingMomentumCalculator alloc] initWithInitialOrigin:origin velocity:velocity documentFrame:contentFrame constrainedClippingOrigin:NSZeroPoint clippingSize:m_scrollExtents.viewportSize tolerance:NSMakeSize(1, 1)]);
    m_initialDestinationOffset = [m_platformMomentumCalculator destinationOrigin];
    return m_platformMomentumCalculator.get();
}

} // namespace WebCore

#endif // PLATFORM(MAC)
