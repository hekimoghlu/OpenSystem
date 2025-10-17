/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 12, 2024.
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
#pragma once

#include "FloatPoint.h"
#include "PlatformWheelEvent.h"
#include "ScrollingEffectsController.h"
#include "Timer.h"
#include "WheelEventTestMonitor.h"
#include <wtf/Forward.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class FloatPoint;
class KeyboardScrollingAnimator;
class PlatformTouchEvent;
class ScrollAnimationSmooth;
class ScrollableArea;
class Scrollbar;
class WheelEventTestMonitor;

struct ScrollExtents;

class ScrollingEffectsControllerTimer;

class ScrollAnimator : private ScrollingEffectsControllerClient {
    WTF_MAKE_TZONE_ALLOCATED(ScrollAnimator);
public:
    static std::unique_ptr<ScrollAnimator> create(ScrollableArea&);

    explicit ScrollAnimator(ScrollableArea&);
    virtual ~ScrollAnimator();

    ScrollableArea& scrollableArea() const { return m_scrollableArea; }

    KeyboardScrollingAnimator *keyboardScrollingAnimator() const final { return m_keyboardScrollingAnimator.get(); }

    enum ScrollBehavior {
        RespectScrollSnap   = 1 << 0,
        NeverAnimate        = 1 << 1,
    };

    // Computes a scroll destination for the given parameters.  Returns false if
    // already at the destination. Otherwise, starts scrolling towards the
    // destination and returns true. Scrolling may be immediate or animated.
    // The base class implementation always scrolls immediately, never animates.
    bool singleAxisScroll(ScrollEventAxis, float delta, OptionSet<ScrollBehavior>);

    WEBCORE_EXPORT bool scrollToPositionWithoutAnimation(const FloatPoint&, ScrollClamping = ScrollClamping::Clamped);
    WEBCORE_EXPORT bool scrollToPositionWithAnimation(const FloatPoint&, ScrollClamping = ScrollClamping::Clamped);

    void retargetRunningAnimation(const FloatPoint& newPosition);

    virtual bool handleWheelEvent(const PlatformWheelEvent&);
    virtual bool processWheelEventForScrollSnap(const PlatformWheelEvent&) { return false; }

    void stopKeyboardScrollAnimation();
    
#if PLATFORM(COCOA)
    virtual void handleWheelEventPhase(PlatformWheelEventPhase) { }
#endif

#if ENABLE(TOUCH_EVENTS)
    virtual bool handleTouchEvent(const PlatformTouchEvent&);
#endif

    void cancelAnimations();

    virtual bool isRubberBandInProgress() const { return false; }

    bool isUserScrollInProgress() const { return m_scrollController.isUserScrollInProgress(); }
    bool isScrollSnapInProgress() const { return m_scrollController.isScrollSnapInProgress(); }
    bool usesScrollSnap() const { return m_scrollController.usesScrollSnap(); }

    void contentsSizeChanged();

    enum NotifyScrollableArea : bool { No, Yes };
    void setCurrentPosition(const FloatPoint&, NotifyScrollableArea = NotifyScrollableArea::No);
    const FloatPoint& currentPosition() const { return m_currentPosition; }

    void setWheelEventTestMonitor(RefPtr<WheelEventTestMonitor>&& testMonitor) { m_wheelEventTestMonitor = testMonitor; }
    WheelEventTestMonitor* wheelEventTestMonitor() const { return m_wheelEventTestMonitor.get(); }

    FloatPoint scrollOffsetAdjustedForSnapping(const FloatPoint& offset, ScrollSnapPointSelectionMethod) const;
    float scrollOffsetAdjustedForSnapping(ScrollEventAxis, const FloatPoint& newOffset, ScrollSnapPointSelectionMethod) const;

    bool activeScrollSnapIndexDidChange() const;
    std::optional<unsigned> activeScrollSnapIndexForAxis(ScrollEventAxis) const;
    void setActiveScrollSnapIndexForAxis(ScrollEventAxis, std::optional<unsigned> index);
    void setSnapOffsetsInfo(const LayoutScrollSnapOffsetsInfo&);
    const LayoutScrollSnapOffsetsInfo* snapOffsetsInfo() const;
    void resnapAfterLayout();

    ScrollAnimationStatus serviceScrollAnimation(MonotonicTime);

protected:
    bool handleSteppedScrolling(const PlatformWheelEvent&);

private:
    void notifyPositionChanged(const FloatSize& delta);

    void updateActiveScrollSnapIndexForOffset();

    FloatPoint offsetFromPosition(const FloatPoint& position) const;
    FloatPoint positionFromOffset(const FloatPoint& offset) const;

    FloatPoint adjustScrollPositionIfNecessary(const FloatPoint&) const;

    // ScrollingEffectsControllerClient.
    std::unique_ptr<ScrollingEffectsControllerTimer> createTimer(Function<void()>&&) final;
    void startAnimationCallback(ScrollingEffectsController&) final;
    void stopAnimationCallback(ScrollingEffectsController&) final;

    FloatPoint scrollOffset() const final;
    float pageScaleFactor() const final;
    ScrollExtents scrollExtents() const final;

    bool allowsHorizontalScrolling() const final;
    bool allowsVerticalScrolling() const final;

    void willStartAnimatedScroll() final;
    void didStopAnimatedScroll() final;

    void immediateScrollBy(const FloatSize&, ScrollClamping = ScrollClamping::Clamped) final;
    void adjustScrollPositionToBoundsIfNecessary() final;

#if HAVE(RUBBER_BANDING)
    IntSize stretchAmount() const final;
    RectEdges<bool> edgePinnedState() const final;
    bool isPinnedOnSide(BoxSide) const final;
#endif

    void deferWheelEventTestCompletionForReason(ScrollingNodeID, WheelEventTestMonitor::DeferReason) const final;
    void removeWheelEventTestCompletionDeferralForReason(ScrollingNodeID, WheelEventTestMonitor::DeferReason) const final;
    ScrollingNodeID scrollingNodeIDForTesting() const final;

#if USE(COORDINATED_GRAPHICS)
    bool scrollAnimationEnabled() const final;
#endif

protected:
    ScrollableArea& m_scrollableArea;
    RefPtr<WheelEventTestMonitor> m_wheelEventTestMonitor;
    ScrollingEffectsController m_scrollController;
    FloatPoint m_currentPosition;

    std::unique_ptr<KeyboardScrollingAnimator> m_keyboardScrollingAnimator;

private:
    bool m_scrollAnimationScheduled { false };
};

} // namespace WebCore

