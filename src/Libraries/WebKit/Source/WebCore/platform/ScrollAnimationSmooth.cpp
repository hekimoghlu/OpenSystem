/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 30, 2024.
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
#include "ScrollAnimationSmooth.h"

#include "FloatPoint.h"
#include "GeometryUtilities.h"
#include "ScrollExtents.h"
#include "ScrollableArea.h"
#include "TimingFunction.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/TextStream.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ScrollAnimationSmooth);

static const float animationSpeed { 1000.0f };
static const Seconds maxAnimationDuration { 200_ms };

ScrollAnimationSmooth::ScrollAnimationSmooth(ScrollAnimationClient& client)
    : ScrollAnimation(Type::Smooth, client)
    , m_timingFunction(CubicBezierTimingFunction::create())
{
}

ScrollAnimationSmooth::~ScrollAnimationSmooth() = default;

bool ScrollAnimationSmooth::startAnimatedScrollToDestination(const FloatPoint& fromOffset, const FloatPoint& destinationOffset)
{
    auto extents = m_client.scrollExtentsForAnimation(*this);

    m_currentOffset = m_startOffset = fromOffset;
    m_destinationOffset = destinationOffset.constrainedBetween(extents.minimumScrollOffset(), extents.maximumScrollOffset());

    if (!isActive() && fromOffset == m_destinationOffset)
        return false;

    m_duration = durationFromDistance(m_destinationOffset - m_startOffset);
    if (!m_duration)
        return false;

    downcast<CubicBezierTimingFunction>(*m_timingFunction).setTimingFunctionPreset(CubicBezierTimingFunction::TimingFunctionPreset::EaseInOut);

    if (!isActive())
        didStart(MonotonicTime::now());

    return true;
}

bool ScrollAnimationSmooth::retargetActiveAnimation(const FloatPoint& newOffset)
{
    if (!isActive())
        return false;

    auto extents = m_client.scrollExtentsForAnimation(*this);

    m_startTime = MonotonicTime::now();
    m_startOffset = m_currentOffset;
    m_destinationOffset = newOffset.constrainedBetween(extents.minimumScrollOffset(), extents.maximumScrollOffset());
    m_duration = durationFromDistance(m_destinationOffset - m_startOffset);
    downcast<CubicBezierTimingFunction>(*m_timingFunction).setTimingFunctionPreset(CubicBezierTimingFunction::TimingFunctionPreset::EaseOut);
    m_timingFunction = CubicBezierTimingFunction::create(CubicBezierTimingFunction::TimingFunctionPreset::EaseOut);

    if (m_currentOffset == m_destinationOffset || !m_duration)
        return false;

    return true;
}

void ScrollAnimationSmooth::updateScrollExtents()
{
    auto extents = m_client.scrollExtentsForAnimation(*this);
    // FIXME: Ideally fix up m_startOffset so m_currentOffset doesn't go backwards.
    m_destinationOffset = m_destinationOffset.constrainedBetween(extents.minimumScrollOffset(), extents.maximumScrollOffset());
}

Seconds ScrollAnimationSmooth::durationFromDistance(const FloatSize& delta) const
{
    float distance = euclidianDistance(delta);
    return std::min(Seconds(distance / animationSpeed), maxAnimationDuration);
}

inline float linearInterpolation(float progress, float a, float b)
{
    return a + progress * (b - a);
}

void ScrollAnimationSmooth::serviceAnimation(MonotonicTime currentTime)
{
    bool animationActive = animateScroll(currentTime);
    m_client.scrollAnimationDidUpdate(*this, m_currentOffset);
    if (!animationActive)
        didEnd();
}

bool ScrollAnimationSmooth::animateScroll(MonotonicTime currentTime)
{
    MonotonicTime endTime = m_startTime + m_duration;
    currentTime = std::min(currentTime, endTime);

    double fractionComplete = (currentTime - m_startTime) / m_duration;
    double progress = m_timingFunction->transformProgress(fractionComplete, m_duration.value());

    m_currentOffset = {
        linearInterpolation(progress, m_startOffset.x(), m_destinationOffset.x()),
        linearInterpolation(progress, m_startOffset.y(), m_destinationOffset.y()),
    };

    return currentTime < endTime;
}

String ScrollAnimationSmooth::debugDescription() const
{
    TextStream textStream;
    textStream << "ScrollAnimationSmooth " << this << " active " << isActive() << " from " << m_startOffset << " to " << m_destinationOffset << " current offset " << currentOffset();
    return textStream.release();
}

} // namespace WebCore
