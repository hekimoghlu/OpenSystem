/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 8, 2023.
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
#include "ScrollAnimator.h"

#include "FloatPoint.h"
#include "KeyboardScrollingAnimator.h"
#include "LayoutSize.h"
#include "PlatformWheelEvent.h"
#include "ScrollExtents.h"
#include "ScrollableArea.h"
#include "ScrollbarsController.h"
#include "ScrollingEffectsController.h"
#include <algorithm>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ScrollAnimator);

#if !PLATFORM(IOS_FAMILY) && !PLATFORM(MAC)
std::unique_ptr<ScrollAnimator> ScrollAnimator::create(ScrollableArea& scrollableArea)
{
    return makeUnique<ScrollAnimator>(scrollableArea);
}
#endif

ScrollAnimator::ScrollAnimator(ScrollableArea& scrollableArea)
    : m_scrollableArea(scrollableArea)
    , m_scrollController(*this)
    , m_keyboardScrollingAnimator(makeUnique<KeyboardScrollingAnimator>(scrollableArea))
{
}

ScrollAnimator::~ScrollAnimator()
{
    m_scrollController.stopAllTimers();
}

bool ScrollAnimator::singleAxisScroll(ScrollEventAxis axis, float scrollDelta, OptionSet<ScrollBehavior> behavior)
{
    m_scrollableArea.scrollbarsController().setScrollbarAnimationsUnsuspendedByUserInteraction(true);

    auto delta = setValueForAxis(FloatSize { }, axis, scrollDelta);

    if (behavior.contains(ScrollBehavior::RespectScrollSnap) && m_scrollController.usesScrollSnap()) {
        auto currentOffset = offsetFromPosition(currentPosition());
        auto newOffset = currentOffset + delta;
        auto velocity = copysignf(1.0f, scrollDelta);
        auto newOffsetOnAxis = m_scrollController.adjustedScrollDestination(axis, newOffset, velocity, valueForAxis(currentOffset, axis));
        newOffset = setValueForAxis(newOffset, axis, newOffsetOnAxis);
        delta = newOffset - currentOffset;
    } else {
        auto newPosition = m_currentPosition + delta;
        newPosition = newPosition.constrainedBetween(scrollableArea().minimumScrollPosition(), scrollableArea().maximumScrollPosition());
        if (newPosition == m_currentPosition)
            return false;

        delta = newPosition - m_currentPosition;
    }

    if (m_scrollableArea.scrollAnimatorEnabled() && !behavior.contains(ScrollBehavior::NeverAnimate)) {
        if (m_scrollController.retargetAnimatedScrollBy(delta))
            return true;

        auto options = ScrollPositionChangeOptions::createProgrammatic();
        options.originalScrollDelta = delta;
        m_scrollableArea.scrollToPositionWithAnimation(m_currentPosition + delta, options);
        return true;
    }

    return scrollToPositionWithoutAnimation(currentPosition() + delta);
}

bool ScrollAnimator::scrollToPositionWithoutAnimation(const FloatPoint& position, ScrollClamping clamping)
{
    FloatPoint currentPosition = this->currentPosition();
    auto adjustedPosition = clamping == ScrollClamping::Clamped ? position.constrainedBetween(scrollableArea().minimumScrollPosition(), scrollableArea().maximumScrollPosition()) : position;

    // FIXME: In some cases on iOS the ScrollableArea position is out of sync with the ScrollAnimator position.
    // When these cases are fixed, this extra check against the ScrollableArea position can be removed.
    if (adjustedPosition == currentPosition && adjustedPosition == scrollableArea().scrollPosition() && !scrollableArea().scrollOriginChanged())
        return false;

    m_scrollController.stopAnimatedScroll();

    setCurrentPosition(adjustedPosition, NotifyScrollableArea::Yes);
    return true;
}

bool ScrollAnimator::scrollToPositionWithAnimation(const FloatPoint& position, ScrollClamping clamping)
{
    auto adjustedPosition = clamping == ScrollClamping::Clamped ? position.constrainedBetween(scrollableArea().minimumScrollPosition(), scrollableArea().maximumScrollPosition()) : position;
    bool positionChanged = adjustedPosition != currentPosition();
    if (!positionChanged && !scrollableArea().scrollOriginChanged())
        return false;

    return m_scrollController.startAnimatedScrollToDestination(offsetFromPosition(m_currentPosition), offsetFromPosition(adjustedPosition));
}

void ScrollAnimator::retargetRunningAnimation(const FloatPoint& newPosition)
{
    ASSERT(scrollableArea().scrollAnimationStatus() == ScrollAnimationStatus::Animating);
    m_scrollController.retargetAnimatedScroll(offsetFromPosition(newPosition));
}

FloatPoint ScrollAnimator::offsetFromPosition(const FloatPoint& position) const
{
    return ScrollableArea::scrollOffsetFromPosition(position, toFloatSize(m_scrollableArea.scrollOrigin()));
}

FloatPoint ScrollAnimator::positionFromOffset(const FloatPoint& offset) const
{
    return ScrollableArea::scrollPositionFromOffset(offset, toFloatSize(m_scrollableArea.scrollOrigin()));
}

bool ScrollAnimator::activeScrollSnapIndexDidChange() const
{
    return m_scrollController.activeScrollSnapIndexDidChange();
}

std::optional<unsigned> ScrollAnimator::activeScrollSnapIndexForAxis(ScrollEventAxis axis) const
{
    return m_scrollController.activeScrollSnapIndexForAxis(axis);
}

void ScrollAnimator::setActiveScrollSnapIndexForAxis(ScrollEventAxis axis, std::optional<unsigned> index)
{
    return m_scrollController.setActiveScrollSnapIndexForAxis(axis, index);
}

void ScrollAnimator::resnapAfterLayout()
{
    m_scrollController.resnapAfterLayout();
}

bool ScrollAnimator::handleWheelEvent(const PlatformWheelEvent& wheelEvent)
{
#if !PLATFORM(MAC)
    m_scrollController.updateGestureInProgressState(wheelEvent);
#endif

    if (processWheelEventForScrollSnap(wheelEvent))
        return false;

    if (m_scrollableArea.hasSteppedScrolling())
        return handleSteppedScrolling(wheelEvent);

    return m_scrollController.handleWheelEvent(wheelEvent);
}

// "Stepped scrolling" is only used by RenderListBox. It's special in that it has no rubberbanding, and scroll deltas respect Scrollbar::pixelStep().
bool ScrollAnimator::handleSteppedScrolling(const PlatformWheelEvent& wheelEvent)
{
    auto* horizontalScrollbar = m_scrollableArea.horizontalScrollbar();
    auto* verticalScrollbar = m_scrollableArea.verticalScrollbar();

    // Accept the event if we have a scrollbar in that direction and can still
    // scroll any further.
    float deltaX = horizontalScrollbar ? wheelEvent.deltaX() : 0;
    float deltaY = verticalScrollbar ? wheelEvent.deltaY() : 0;

    bool handled = false;

    IntSize maxForwardScrollDelta = m_scrollableArea.maximumScrollPosition() - m_scrollableArea.scrollPosition();
    IntSize maxBackwardScrollDelta = m_scrollableArea.scrollPosition() - m_scrollableArea.minimumScrollPosition();
    if ((deltaX < 0 && maxForwardScrollDelta.width() > 0)
        || (deltaX > 0 && maxBackwardScrollDelta.width() > 0)
        || (deltaY < 0 && maxForwardScrollDelta.height() > 0)
        || (deltaY > 0 && maxBackwardScrollDelta.height() > 0)) {
        handled = true;

        OptionSet<ScrollBehavior> behavior = { ScrollBehavior::RespectScrollSnap };
        if (wheelEvent.hasPreciseScrollingDeltas())
            behavior.add(ScrollBehavior::NeverAnimate);

        if (deltaY) {
            if (wheelEvent.granularity() == ScrollByPageWheelEvent)
                deltaY = std::copysign(Scrollbar::pageStepDelta(m_scrollableArea.visibleHeight()), deltaY);

            auto scrollDelta = verticalScrollbar->pixelStep() * -deltaY; // Wheel deltas are reversed from scrolling direction.
            singleAxisScroll(ScrollEventAxis::Vertical, scrollDelta, behavior);
        }

        if (deltaX) {
            if (wheelEvent.granularity() == ScrollByPageWheelEvent)
                deltaX = std::copysign(Scrollbar::pageStepDelta(m_scrollableArea.visibleWidth()), deltaX);

            auto scrollDelta = horizontalScrollbar->pixelStep() * -deltaX; // Wheel deltas are reversed from scrolling direction.
            singleAxisScroll(ScrollEventAxis::Horizontal, scrollDelta, behavior);
        }
    }
    return handled;
}

void ScrollAnimator::stopKeyboardScrollAnimation()
{
    m_scrollController.stopKeyboardScrolling();
}

#if ENABLE(TOUCH_EVENTS)
bool ScrollAnimator::handleTouchEvent(const PlatformTouchEvent&)
{
    return false;
}
#endif

static void notifyScrollAnchoringControllerOfScroll(ScrollableArea& scrollableArea)
{
    scrollableArea.invalidateScrollAnchoringElement();
    scrollableArea.updateScrollAnchoringElement();
}

void ScrollAnimator::setCurrentPosition(const FloatPoint& position, NotifyScrollableArea notify)
{
    // FIXME: An early return here if the position is not changing triggers test failures because of adjustForIOSCaretWhenScrolling()
    // code in RenderLayerScrollableArea. We can early return when webkit.org/b/230454 is fixed.
    auto delta = position - m_currentPosition;
    m_currentPosition = position;
    
    if (notify == NotifyScrollableArea::Yes)
        notifyPositionChanged(delta);
    else
        notifyScrollAnchoringControllerOfScroll(m_scrollableArea);

    updateActiveScrollSnapIndexForOffset();
}

void ScrollAnimator::updateActiveScrollSnapIndexForOffset()
{
    m_scrollController.updateActiveScrollSnapIndexForClientOffset();
}

void ScrollAnimator::notifyPositionChanged(const FloatSize& delta)
{
    m_scrollableArea.scrollbarsController().notifyContentAreaScrolled(delta);
    m_scrollableArea.setScrollPositionFromAnimation(roundedIntPoint(m_currentPosition));
    m_scrollController.scrollPositionChanged();
}

void ScrollAnimator::setSnapOffsetsInfo(const LayoutScrollSnapOffsetsInfo& info)
{
    m_scrollController.setSnapOffsetsInfo(info);
}

const LayoutScrollSnapOffsetsInfo* ScrollAnimator::snapOffsetsInfo() const
{
    return m_scrollController.snapOffsetsInfo();
}

FloatPoint ScrollAnimator::scrollOffset() const
{
    return m_scrollableArea.scrollOffsetFromPosition(roundedIntPoint(currentPosition()));
}

bool ScrollAnimator::allowsHorizontalScrolling() const
{
    return m_scrollableArea.allowsHorizontalScrolling();
}

bool ScrollAnimator::allowsVerticalScrolling() const
{
    return m_scrollableArea.allowsVerticalScrolling();
}

void ScrollAnimator::willStartAnimatedScroll()
{
    m_scrollableArea.setScrollAnimationStatus(ScrollAnimationStatus::Animating);
}

void ScrollAnimator::didStopAnimatedScroll()
{
    m_scrollableArea.setScrollAnimationStatus(ScrollAnimationStatus::NotAnimating);
    m_scrollableArea.animatedScrollDidEnd();
}

#if HAVE(RUBBER_BANDING)
IntSize ScrollAnimator::stretchAmount() const
{
    return m_scrollableArea.overhangAmount();
}

RectEdges<bool> ScrollAnimator::edgePinnedState() const
{
    return m_scrollableArea.edgePinnedState();
}

bool ScrollAnimator::isPinnedOnSide(BoxSide side) const
{
    return m_scrollableArea.isPinnedOnSide(side);
}

#endif

void ScrollAnimator::adjustScrollPositionToBoundsIfNecessary()
{
    auto previousClamping = m_scrollableArea.scrollClamping();
    m_scrollableArea.setScrollClamping(ScrollClamping::Clamped);

    auto currentScrollPosition = m_scrollableArea.scrollPosition();
    auto constrainedPosition = m_scrollableArea.constrainedScrollPosition(currentScrollPosition);
    immediateScrollBy(constrainedPosition - currentScrollPosition);

    m_scrollableArea.setScrollClamping(previousClamping);
}

FloatPoint ScrollAnimator::adjustScrollPositionIfNecessary(const FloatPoint& position) const
{
    if (m_scrollableArea.scrollClamping() == ScrollClamping::Unclamped)
        return position;

    return m_scrollableArea.constrainedScrollPosition(ScrollPosition(position));
}

void ScrollAnimator::immediateScrollBy(const FloatSize& delta, ScrollClamping clamping)
{
    auto previousClamping = m_scrollableArea.scrollClamping();
    m_scrollableArea.setScrollClamping(clamping);

    auto currentPosition = this->currentPosition();
    auto newPosition = adjustScrollPositionIfNecessary(currentPosition + delta);
    if (newPosition != currentPosition)
        setCurrentPosition(newPosition, NotifyScrollableArea::Yes);

    m_scrollableArea.setScrollClamping(previousClamping);
}

ScrollExtents ScrollAnimator::scrollExtents() const
{
    return {
        m_scrollableArea.totalContentsSize(),
        m_scrollableArea.visibleSize()
    };
}

float ScrollAnimator::pageScaleFactor() const
{
    return m_scrollableArea.pageScaleFactor();
}

std::unique_ptr<ScrollingEffectsControllerTimer> ScrollAnimator::createTimer(Function<void()>&& function)
{
    return makeUnique<ScrollingEffectsControllerTimer>(RunLoop::current(), [function = WTFMove(function), weakScrollableArea = WeakPtr { m_scrollableArea }] {
        if (!weakScrollableArea)
            return;
        function();
    });
}

void ScrollAnimator::startAnimationCallback(ScrollingEffectsController&)
{
    if (!m_scrollAnimationScheduled) {
        m_scrollAnimationScheduled = true;
        m_scrollableArea.didStartScrollAnimation();
    }
}

void ScrollAnimator::stopAnimationCallback(ScrollingEffectsController&)
{
    m_scrollAnimationScheduled = false;
}

void ScrollAnimator::deferWheelEventTestCompletionForReason(ScrollingNodeID identifier, WheelEventTestMonitor::DeferReason reason) const
{
    if (!m_wheelEventTestMonitor)
        return;

    m_wheelEventTestMonitor->deferForReason(identifier, reason);
}

void ScrollAnimator::removeWheelEventTestCompletionDeferralForReason(ScrollingNodeID identifier, WheelEventTestMonitor::DeferReason reason) const
{
    if (!m_wheelEventTestMonitor)
        return;
    
    m_wheelEventTestMonitor->removeDeferralForReason(identifier, reason);
}

#if USE(COORDINATED_GRAPHICS)
bool ScrollAnimator::scrollAnimationEnabled() const
{
    return m_scrollableArea.scrollAnimatorEnabled();
}
#endif

void ScrollAnimator::cancelAnimations()
{
    m_scrollController.stopAnimatedScroll();
    m_scrollableArea.scrollbarsController().cancelAnimations();
}

void ScrollAnimator::contentsSizeChanged()
{
    m_scrollController.contentsSizeChanged();
}

FloatPoint ScrollAnimator::scrollOffsetAdjustedForSnapping(const FloatPoint& offset, ScrollSnapPointSelectionMethod method) const
{
    if (!m_scrollController.usesScrollSnap())
        return offset;

    return {
        scrollOffsetAdjustedForSnapping(ScrollEventAxis::Horizontal, offset, method),
        scrollOffsetAdjustedForSnapping(ScrollEventAxis::Vertical, offset, method)
    };
}

float ScrollAnimator::scrollOffsetAdjustedForSnapping(ScrollEventAxis axis, const FloatPoint& newOffset, ScrollSnapPointSelectionMethod method) const
{
    if (!m_scrollController.usesScrollSnap())
        return axis == ScrollEventAxis::Horizontal ? newOffset.x() : newOffset.y();

    std::optional<float> originalOffset;
    float velocityInScrollAxis = 0.;
    if (method == ScrollSnapPointSelectionMethod::Directional) {
        FloatSize scrollOrigin = toFloatSize(m_scrollableArea.scrollOrigin());
        auto currentOffset = ScrollableArea::scrollOffsetFromPosition(this->currentPosition(), scrollOrigin);
        auto velocity = newOffset - currentOffset;
        originalOffset = axis == ScrollEventAxis::Horizontal ? currentOffset.x() : currentOffset.y();
        velocityInScrollAxis = axis == ScrollEventAxis::Horizontal ? velocity.width() : velocity.height();
    }

    return m_scrollController.adjustedScrollDestination(axis, newOffset, velocityInScrollAxis, originalOffset);
}

ScrollAnimationStatus ScrollAnimator::serviceScrollAnimation(MonotonicTime time)
{
    if (m_scrollAnimationScheduled)
        m_scrollController.animationCallback(time);
    return m_scrollAnimationScheduled ? ScrollAnimationStatus::Animating : ScrollAnimationStatus::NotAnimating;
}

ScrollingNodeID ScrollAnimator::scrollingNodeIDForTesting() const
{
    return m_scrollableArea.scrollingNodeIDForTesting();
}


} // namespace WebCore
