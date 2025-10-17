/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 12, 2022.
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
#import "ScrollAnimatorMac.h"

#if PLATFORM(MAC)

#import "Gradient.h"
#import "GraphicsLayer.h"
#import "Logging.h"
#import "PlatformWheelEvent.h"
#import "ScrollView.h"
#import "ScrollableArea.h"
#import "ScrollbarsController.h"
#import <wtf/TZoneMallocInlines.h>
#import <wtf/text/TextStream.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ScrollAnimatorMac);

std::unique_ptr<ScrollAnimator> ScrollAnimator::create(ScrollableArea& scrollableArea)
{
    return makeUnique<ScrollAnimatorMac>(scrollableArea);
}

ScrollAnimatorMac::ScrollAnimatorMac(ScrollableArea& scrollableArea)
    : ScrollAnimator(scrollableArea)
{
}

ScrollAnimatorMac::~ScrollAnimatorMac() = default;

bool ScrollAnimatorMac::isRubberBandInProgress() const
{
    return m_scrollController.isRubberBandInProgress();
}

void ScrollAnimatorMac::handleWheelEventPhase(PlatformWheelEventPhase phase)
{
    LOG_WITH_STREAM(OverlayScrollbars, stream << "ScrollAnimatorMac " << this << " scrollableArea " << m_scrollableArea << " handleWheelEventPhase " << phase);

    // FIXME: Need to ensure we get PlatformWheelEventPhase::Ended.
    if (phase == PlatformWheelEventPhase::Began)
        m_scrollableArea.scrollbarsController().didBeginScrollGesture();
    else if (phase == PlatformWheelEventPhase::Ended || phase == PlatformWheelEventPhase::Cancelled)
        m_scrollableArea.scrollbarsController().didEndScrollGesture();
    else if (phase == PlatformWheelEventPhase::MayBegin)
        m_scrollableArea.scrollbarsController().mayBeginScrollGesture();
}

bool ScrollAnimatorMac::handleWheelEvent(const PlatformWheelEvent& wheelEvent)
{
    m_scrollableArea.scrollbarsController().setScrollbarAnimationsUnsuspendedByUserInteraction(true);
    m_scrollController.updateGestureInProgressState(wheelEvent);

    // Events in the PlatformWheelEventPhase::MayBegin phase have no deltas, and therefore never passes through the scroll handling logic below.
    // This causes us to return with an 'unhandled' return state, even though this event was successfully processed.
    //
    // We receive at least one PlatformWheelEventPhase::MayBegin when starting main-thread scrolling (see LocalFrameView::wheelEvent), which can
    // fool the scrolling thread into attempting to handle the scroll, unless we treat the event as handled here.
    if (wheelEvent.phase() == PlatformWheelEventPhase::MayBegin)
        return true;

    bool didHandleEvent = ScrollAnimator::handleWheelEvent(wheelEvent);
    if (didHandleEvent || wheelEvent.delta().isZero())
        handleWheelEventPhase(wheelEvent.phase());

    return didHandleEvent;
}

static bool gestureShouldBeginSnap(const PlatformWheelEvent& wheelEvent, ScrollEventAxis axis, const LayoutScrollSnapOffsetsInfo* offsetInfo)
{
    if (!offsetInfo)
        return false;

    if (offsetInfo->offsetsForAxis(axis).isEmpty())
        return false;

    if (wheelEvent.phase() != PlatformWheelEventPhase::Ended && !wheelEvent.isEndOfMomentumScroll())
        return false;

    return true;
}

bool ScrollAnimatorMac::allowsVerticalStretching(const PlatformWheelEvent& wheelEvent) const
{
    if (m_scrollableArea.verticalOverscrollBehavior() == OverscrollBehavior::None)
        return false;
    
    switch (m_scrollableArea.verticalScrollElasticity()) {
    case ScrollElasticity::Automatic: {
        Scrollbar* hScroller = m_scrollableArea.horizontalScrollbar();
        Scrollbar* vScroller = m_scrollableArea.verticalScrollbar();
        bool scrollbarsAllowStretching = ((vScroller && vScroller->enabled()) || (!hScroller || !hScroller->enabled()));
        auto relevantSide = ScrollableArea::targetSideForScrollDelta(-wheelEvent.delta(), ScrollEventAxis::Vertical);
        bool eventPreventsStretching = m_scrollableArea.hasScrollableOrRubberbandableAncestor() && wheelEvent.isGestureStart() && relevantSide && m_scrollableArea.isPinnedOnSide(*relevantSide);
        if (!eventPreventsStretching)
            eventPreventsStretching = gestureShouldBeginSnap(wheelEvent, ScrollEventAxis::Vertical, m_scrollableArea.snapOffsetsInfo());
        return scrollbarsAllowStretching && !eventPreventsStretching;
    }
    case ScrollElasticity::None:
        return false;
    case ScrollElasticity::Allowed:
        return true;
    }

    ASSERT_NOT_REACHED();
    return false;
}

bool ScrollAnimatorMac::allowsHorizontalStretching(const PlatformWheelEvent& wheelEvent) const
{
    if (m_scrollableArea.horizontalOverscrollBehavior() == OverscrollBehavior::None)
        return false;
    
    switch (m_scrollableArea.horizontalScrollElasticity()) {
    case ScrollElasticity::Automatic: {
        Scrollbar* hScroller = m_scrollableArea.horizontalScrollbar();
        Scrollbar* vScroller = m_scrollableArea.verticalScrollbar();
        bool scrollbarsAllowStretching = ((hScroller && hScroller->enabled()) || (!vScroller || !vScroller->enabled()));
        auto relevantSide = ScrollableArea::targetSideForScrollDelta(-wheelEvent.delta(), ScrollEventAxis::Horizontal);
        bool eventPreventsStretching = m_scrollableArea.hasScrollableOrRubberbandableAncestor() && wheelEvent.isGestureStart() && relevantSide && m_scrollableArea.isPinnedOnSide(*relevantSide);
        if (!eventPreventsStretching)
            eventPreventsStretching = gestureShouldBeginSnap(wheelEvent, ScrollEventAxis::Horizontal, m_scrollableArea.snapOffsetsInfo());
        return scrollbarsAllowStretching && !eventPreventsStretching;
    }
    case ScrollElasticity::None:
        return false;
    case ScrollElasticity::Allowed:
        return true;
    }

    ASSERT_NOT_REACHED();
    return false;
}

bool ScrollAnimatorMac::shouldRubberBandOnSide(BoxSide) const
{
    return false;
}

bool ScrollAnimatorMac::processWheelEventForScrollSnap(const PlatformWheelEvent& wheelEvent)
{
    return m_scrollController.processWheelEventForScrollSnap(wheelEvent);
}

} // namespace WebCore

#endif // PLATFORM(MAC)
