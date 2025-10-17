/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 18, 2024.
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
#import "ScrollAnimatorIOS.h"

#if PLATFORM(IOS_FAMILY)

#import "LocalFrame.h"
#import "RenderLayer.h"
#import "ScrollableArea.h"
#import "ScrollingEffectsController.h"
#import <wtf/TZoneMallocInlines.h>

#if ENABLE(TOUCH_EVENTS)
#import "PlatformTouchEventIOS.h"
#endif

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ScrollAnimatorIOS);

std::unique_ptr<ScrollAnimator> ScrollAnimator::create(ScrollableArea& scrollableArea)
{
    return makeUnique<ScrollAnimatorIOS>(scrollableArea);
}

ScrollAnimatorIOS::ScrollAnimatorIOS(ScrollableArea& scrollableArea)
    : ScrollAnimator(scrollableArea)
{
}

ScrollAnimatorIOS::~ScrollAnimatorIOS()
{
}

#if ENABLE(TOUCH_EVENTS)
bool ScrollAnimatorIOS::handleTouchEvent(const PlatformTouchEvent& touchEvent)
{
    if (touchEvent.type() == PlatformEvent::Type::TouchStart && touchEvent.touchCount() == 1) {
        m_firstTouchPoint = touchEvent.touchLocationInRootViewAtIndex(0);
        m_lastTouchPoint = m_firstTouchPoint;
        m_inTouchSequence = true;
        m_committedToScrollAxis = false;
        m_startedScroll = false;
        m_touchScrollAxisLatch = AxisLatchNotComputed;
        // Never claim to have handled the TouchStart, because that will kill default scrolling behavior.
        return false;
    }

    if (!m_inTouchSequence)
        return false;

    if (touchEvent.type() == PlatformEvent::Type::TouchEnd || touchEvent.type() == PlatformEvent::Type::TouchCancel) {
        m_inTouchSequence = false;
        m_scrollableAreaForTouchSequence = 0;
        if (m_startedScroll)
            scrollableArea().didEndScroll();
        return false;
    }

    // If a second touch appears, assume that the user is trying to zoom, and bail on the scrolling sequence.
    // FIXME: if that second touch is inside the scrollable area, should we keep scrolling?
    if (touchEvent.touchCount() != 1) {
        m_inTouchSequence = false;
        m_scrollableAreaForTouchSequence = 0;
        if (m_startedScroll)
            scrollableArea().didEndScroll();
        return false;
    }
    
    IntPoint currentPoint = touchEvent.touchLocationInRootViewAtIndex(0);

    IntSize touchDelta = m_lastTouchPoint - currentPoint;
    m_lastTouchPoint = currentPoint;

    if (!m_scrollableAreaForTouchSequence)
        determineScrollableAreaForTouchSequence(touchDelta);

    if (!m_committedToScrollAxis) {
        auto scrollSize = m_scrollableArea.maximumScrollPosition() - m_scrollableArea.minimumScrollPosition();
        bool horizontallyScrollable = scrollSize.width();
        bool verticallyScrollable = scrollSize.height();

        if (!horizontallyScrollable && !verticallyScrollable)
            return false;

        IntSize deltaFromStart = m_firstTouchPoint - currentPoint;
    
        const int latchAxisMovementThreshold = 10;
        if (!horizontallyScrollable && verticallyScrollable) {
            m_touchScrollAxisLatch = AxisLatchVertical;
            if (std::abs(deltaFromStart.height()) >= latchAxisMovementThreshold)
                m_committedToScrollAxis = true;
        } else if (horizontallyScrollable && !verticallyScrollable) {
            m_touchScrollAxisLatch = AxisLatchHorizontal;
            if (std::abs(deltaFromStart.width()) >= latchAxisMovementThreshold)
                m_committedToScrollAxis = true;
        } else {
            m_committedToScrollAxis = true;

            if (m_touchScrollAxisLatch == AxisLatchNotComputed) {
                const float lockAngleDegrees = 20;
                if (deltaFromStart.width() && deltaFromStart.height()) {
                    float dragAngle = atanf(static_cast<float>(std::abs(deltaFromStart.height())) / std::abs(deltaFromStart.width()));
                    if (dragAngle <= deg2rad(lockAngleDegrees))
                        m_touchScrollAxisLatch = AxisLatchHorizontal;
                    else if (dragAngle >= deg2rad(90 - lockAngleDegrees))
                        m_touchScrollAxisLatch = AxisLatchVertical;
                }
            }
        }
        
        if (!m_committedToScrollAxis)
            return false;
    }

    bool handled = false;
    
    // Horizontal
    if (m_touchScrollAxisLatch != AxisLatchVertical) {
        int delta = touchDelta.width();
        handled |= m_scrollableAreaForTouchSequence->scroll(delta < 0 ? ScrollDirection::ScrollLeft : ScrollDirection::ScrollRight, ScrollGranularity::Pixel, std::abs(delta));
    }
    
    // Vertical
    if (m_touchScrollAxisLatch != AxisLatchHorizontal) {
        int delta = touchDelta.height();
        handled |= m_scrollableAreaForTouchSequence->scroll(delta < 0 ? ScrollDirection::ScrollUp : ScrollDirection::ScrollDown, ScrollGranularity::Pixel, std::abs(delta));
    }
    
    // Return false until we manage to scroll at all, and then keep returning true until the gesture ends.
    if (!m_startedScroll) {
        if (!handled)
            return false;
        m_startedScroll = true;
        scrollableArea().didStartScroll();
    } else if (handled)
        scrollableArea().didUpdateScroll();
    
    return true;
}

void ScrollAnimatorIOS::determineScrollableAreaForTouchSequence(const IntSize& scrollDelta)
{
    ASSERT(!m_scrollableAreaForTouchSequence);

    auto horizontalEdge = ScrollableArea::targetSideForScrollDelta(scrollDelta, ScrollEventAxis::Horizontal);
    auto verticalEdge = ScrollableArea::targetSideForScrollDelta(scrollDelta, ScrollEventAxis::Vertical);

    auto* scrollableArea = &m_scrollableArea;
    while (true) {
        if (verticalEdge && !scrollableArea->isPinnedOnSide(*verticalEdge))
            break;

        if (horizontalEdge && !scrollableArea->isPinnedOnSide(*horizontalEdge))
            break;

        auto* enclosingArea = scrollableArea->enclosingScrollableArea();
        if (!enclosingArea)
            break;

        scrollableArea = enclosingArea;
    }

    ASSERT(scrollableArea);
    m_scrollableAreaForTouchSequence = scrollableArea;
}
#endif

} // namespace WebCore

#endif // PLATFORM(IOS_FAMILY)
