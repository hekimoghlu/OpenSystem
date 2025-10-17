/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 21, 2025.
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
#import "ScrollAnimationRubberBand.h"

#if HAVE(RUBBER_BANDING)

#import "FloatPoint.h"
#import "GeometryUtilities.h"
#import <pal/spi/mac/NSScrollViewSPI.h>
#import <wtf/TZoneMallocInlines.h>

static float elasticDeltaForTimeDelta(float initialPosition, float initialVelocity, Seconds elapsedTime)
{
    return _NSElasticDeltaForTimeDelta(initialPosition, initialVelocity, elapsedTime.seconds());
}

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ScrollAnimationRubberBand);

static inline float roundTowardZero(float num)
{
    return num > 0 ? ceilf(num - 0.5f) : floorf(num + 0.5f);
}

static inline float roundToDevicePixelTowardZero(float num)
{
    float roundedNum = roundf(num);
    if (std::abs(num - roundedNum) < 0.125)
        num = roundedNum;

    return roundTowardZero(num);
}

ScrollAnimationRubberBand::ScrollAnimationRubberBand(ScrollAnimationClient& client)
    : ScrollAnimation(Type::RubberBand, client)
{
}

ScrollAnimationRubberBand::~ScrollAnimationRubberBand() = default;

bool ScrollAnimationRubberBand::startRubberBandAnimation(const FloatSize& initialVelocity, const FloatSize& initialOverscroll)
{
    m_initialVelocity = initialVelocity;
    m_initialOverscroll = initialOverscroll;

    didStart(MonotonicTime::now());
    return true;
}

bool ScrollAnimationRubberBand::retargetActiveAnimation(const FloatPoint&)
{
    return false;
}

void ScrollAnimationRubberBand::updateScrollExtents()
{
    // FIXME: If we're rubberbanding at the bottom and the content size changes we should fix up m_targetOffset.
}

void ScrollAnimationRubberBand::serviceAnimation(MonotonicTime currentTime)
{
    auto elapsedTime = timeSinceStart(currentTime);

    // This is very similar to ScrollingMomentumCalculator logic, but I wasn't able to get to ScrollingMomentumCalculator to
    // give the correct behavior when starting a rubberband with initial velocity (i.e. bouncing).

    auto rubberBandOffset = FloatSize {
        roundToDevicePixelTowardZero(elasticDeltaForTimeDelta(m_initialOverscroll.width(), -m_initialVelocity.width(), elapsedTime)),
        roundToDevicePixelTowardZero(elasticDeltaForTimeDelta(m_initialOverscroll.height(), -m_initialVelocity.height(), elapsedTime))
    };

    // We might be rubberbanding away from an edge and back, so wait a frame or two before checking for completion.
    bool animationComplete = rubberBandOffset.isZero() && elapsedTime > 24_ms;
    
    auto scrollDelta = rubberBandOffset - m_client.overscrollAmount(*this);
    m_currentOffset = m_client.scrollOffset(*this) + scrollDelta;

    m_client.scrollAnimationDidUpdate(*this, m_currentOffset);

    if (animationComplete)
        didEnd();
}

String ScrollAnimationRubberBand::debugDescription() const
{
    TextStream textStream;
    textStream << "ScrollAnimationRubberBand " << this << " active " << isActive() << " initial velocity " << m_initialVelocity << " initial overscroll " << m_initialOverscroll;
    return textStream.release();
}

} // namespace WebCore

#endif // HAVE(RUBBER_BANDING)
