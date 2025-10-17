/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 20, 2025.
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
#include "ScrollAnimationMomentum.h"

#include "Logging.h"
#include "ScrollingMomentumCalculator.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/TextStream.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ScrollAnimationMomentum);

ScrollAnimationMomentum::ScrollAnimationMomentum(ScrollAnimationClient& client)
    : ScrollAnimation(Type::Momentum, client)
{
}

ScrollAnimationMomentum::~ScrollAnimationMomentum() = default;

bool ScrollAnimationMomentum::startAnimatedScrollWithInitialVelocity(const FloatPoint& initialOffset, const FloatSize& initialVelocity, const FloatSize& initialDelta, const Function<FloatPoint(const FloatPoint&)>& destinationModifier)
{
    auto extents = m_client.scrollExtentsForAnimation(*this);
    m_currentOffset = initialOffset;

    m_momentumCalculator = ScrollingMomentumCalculator::create(extents, initialOffset, initialDelta, initialVelocity);
    auto destinationScrollOffset = m_momentumCalculator->destinationScrollOffset();

    if (destinationModifier) {
        auto modifiedOffset = destinationModifier(destinationScrollOffset);
        if (modifiedOffset != destinationScrollOffset) {
            LOG_WITH_STREAM(ScrollAnimations, stream << "ScrollAnimationMomentum " << this << " startAnimatedScrollWithInitialVelocity - predicted offset " << destinationScrollOffset << " modified to " << modifiedOffset);
            destinationScrollOffset = modifiedOffset;
            m_momentumCalculator->setRetargetedScrollOffset(destinationScrollOffset);
        }
    }

    LOG(ScrollAnimations, "ScrollAnimationMomentum::startAnimatedScrollWithInitialVelocity: velocity %.2f,%.2f from %.2f,%.2f to %.2f,%.2f",
        initialVelocity.width(), initialVelocity.height(), initialOffset.x(), initialOffset.y(), destinationScrollOffset.x(), destinationScrollOffset.y());

    if (destinationScrollOffset == initialOffset) {
        m_momentumCalculator = nullptr;
        return false;
    }

    didStart(MonotonicTime::now());
    return true;
}

bool ScrollAnimationMomentum::retargetActiveAnimation(const FloatPoint& newDestination)
{
    if (m_momentumCalculator && isActive()) {
        m_momentumCalculator->setRetargetedScrollOffset(newDestination);
        auto newDuration = m_momentumCalculator->animationDuration();
        bool animationComplete = newDuration > 0_s;
        if (animationComplete)
            didEnd();

        return !animationComplete;
    }

    return false;
}

void ScrollAnimationMomentum::stop()
{
    LOG(ScrollAnimations, "ScrollAnimationMomentum::stop: offset %.2f,%.2f", m_currentOffset.x(), m_currentOffset.y());

    m_momentumCalculator = nullptr;
    ScrollAnimation::stop();
}

void ScrollAnimationMomentum::serviceAnimation(MonotonicTime currentTime)
{
    if (!m_momentumCalculator) {
        ASSERT_NOT_REACHED();
        return;
    }

    auto elapsedTime = timeSinceStart(currentTime);
    bool animationComplete = elapsedTime >= m_momentumCalculator->animationDuration();
    m_currentOffset = m_momentumCalculator->scrollOffsetAfterElapsedTime(elapsedTime);

    m_client.scrollAnimationDidUpdate(*this, m_currentOffset);

    LOG(ScrollAnimations, "ScrollAnimationMomentum::serviceAnimation: offset %.2f,%.2f complete %d", m_currentOffset.x(), m_currentOffset.y(), animationComplete);

    if (animationComplete)
        didEnd();
}

void ScrollAnimationMomentum::updateScrollExtents()
{
    auto extents = m_client.scrollExtentsForAnimation(*this);
    auto predictedScrollOffset = m_momentumCalculator->destinationScrollOffset();
    auto constrainedOffset = predictedScrollOffset.constrainedBetween(extents.minimumScrollOffset(), extents.maximumScrollOffset());
    if (constrainedOffset != predictedScrollOffset)
        retargetActiveAnimation(constrainedOffset);
}

String ScrollAnimationMomentum::debugDescription() const
{
    TextStream textStream;
    textStream << "ScrollAnimationMomentum " << this << " active " << isActive() << " destination " << (m_momentumCalculator ? m_momentumCalculator->destinationScrollOffset() : FloatPoint()) << " current offset " << currentOffset();
    return textStream.release();
}

} // namespace WebCore
