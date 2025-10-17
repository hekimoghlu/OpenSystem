/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 9, 2024.
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

#include "ScrollingTreeScrollingNodeDelegate.h"

#if ENABLE(ASYNC_SCROLLING) && ENABLE(SCROLLING_THREAD)

#include "ScrollingEffectsController.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class FloatPoint;
class FloatSize;
class IntPoint;
class ScrollingStateScrollingNode;
class ScrollingTreeScrollingNode;
class ScrollingTree;

class ThreadedScrollingTreeScrollingNodeDelegate : public ScrollingTreeScrollingNodeDelegate, private ScrollingEffectsControllerClient {
    WTF_MAKE_TZONE_ALLOCATED(ThreadedScrollingTreeScrollingNodeDelegate);
public:
    void updateSnapScrollState();

protected:
    explicit ThreadedScrollingTreeScrollingNodeDelegate(ScrollingTreeScrollingNode&);
    virtual ~ThreadedScrollingTreeScrollingNodeDelegate();

    void updateUserScrollInProgressForEvent(const PlatformWheelEvent&);

    void updateFromStateNode(const ScrollingStateScrollingNode&) override;

    bool startAnimatedScrollToPosition(FloatPoint) override;
    void stopAnimatedScroll() override;
    void serviceScrollAnimation(MonotonicTime) override;

    // ScrollingEffectsControllerClient.
    std::unique_ptr<ScrollingEffectsControllerTimer> createTimer(Function<void()>&&) override;
    void startAnimationCallback(ScrollingEffectsController&) override;
    void stopAnimationCallback(ScrollingEffectsController&) override;

    bool allowsHorizontalScrolling() const override;
    bool allowsVerticalScrolling() const override;

    void immediateScrollBy(const FloatSize&, ScrollClamping = ScrollClamping::Clamped) override;
    void adjustScrollPositionToBoundsIfNecessary() override;

    bool scrollPositionIsNotRubberbandingEdge(const FloatPoint&) const;

    FloatPoint scrollOffset() const override;
    float pageScaleFactor() const override;

    void willStartAnimatedScroll() override;
    void didStopAnimatedScroll() override;
    void willStartWheelEventScroll() override;
    void didStopWheelEventScroll() override;
    void willStartScrollSnapAnimation() override;
    void didStopScrollSnapAnimation() override;

    ScrollExtents scrollExtents() const override;

    void deferWheelEventTestCompletionForReason(ScrollingNodeID, WheelEventTestMonitor::DeferReason) const override;
    void removeWheelEventTestCompletionDeferralForReason(ScrollingNodeID, WheelEventTestMonitor::DeferReason) const override;

    ScrollingNodeID scrollingNodeIDForTesting() const final;

    FloatPoint adjustedScrollPosition(const FloatPoint&) const override;

    void handleKeyboardScrollRequest(const RequestedKeyboardScrollData&) override;

    ScrollingEffectsController m_scrollController;
};

} // namespace WebCore

#endif // ENABLE(ASYNC_SCROLLING) && ENABLE(SCROLLING_THREAD)
