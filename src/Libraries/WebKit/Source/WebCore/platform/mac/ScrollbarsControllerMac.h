/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 29, 2023.
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

#if PLATFORM(MAC)

#include "IntRect.h"
#include "ScrollbarsController.h"
#include "Timer.h"
#include <wtf/CheckedPtr.h>
#include <wtf/RetainPtr.h>
#include <wtf/TZoneMalloc.h>

OBJC_CLASS WebScrollerImpPairDelegate;
OBJC_CLASS WebScrollerImpDelegate;

typedef id ScrollerImpPair;

namespace WebCore {

class WheelEventTestMonitor;

class ScrollbarsControllerMac final : public ScrollbarsController, public CanMakeCheckedPtr<ScrollbarsControllerMac> {
    WTF_MAKE_TZONE_ALLOCATED(ScrollbarsControllerMac);
    WTF_MAKE_NONCOPYABLE(ScrollbarsControllerMac);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(ScrollbarsControllerMac);
public:
    explicit ScrollbarsControllerMac(ScrollableArea&);
    ~ScrollbarsControllerMac();

    bool isScrollbarsControllerMac() const final { return true; }

    void notifyContentAreaScrolled(const FloatSize& delta) final;

    void cancelAnimations() final;

    void didBeginScrollGesture() final;
    void didEndScrollGesture() final;
    void mayBeginScrollGesture() final;

    void contentAreaWillPaint() const final;
    void mouseEnteredContentArea() final;
    void mouseExitedContentArea() final;
    void mouseMovedInContentArea() final;
    void mouseEnteredScrollbar(Scrollbar*) const final;
    void mouseExitedScrollbar(Scrollbar*) const final;
    void mouseIsDownInScrollbar(Scrollbar*, bool) const final;
    void willStartLiveResize() final;
    void contentsSizeChanged() const final;
    void willEndLiveResize() final;
    void contentAreaDidShow() final;
    void contentAreaDidHide() final;

    void lockOverlayScrollbarStateToHidden(bool shouldLockState) final;
    bool scrollbarsCanBeActive() const final;

    void didAddVerticalScrollbar(Scrollbar*) final;
    void willRemoveVerticalScrollbar(Scrollbar*) final;
    void didAddHorizontalScrollbar(Scrollbar*) final;
    void willRemoveHorizontalScrollbar(Scrollbar*) final;

    void invalidateScrollbarPartLayers(Scrollbar*) final;

    void verticalScrollbarLayerDidChange() final;
    void horizontalScrollbarLayerDidChange() final;

    bool shouldScrollbarParticipateInHitTesting(Scrollbar*) final;

    String horizontalScrollbarStateForTesting() const final;
    String verticalScrollbarStateForTesting() const final;


    // Public to be callable from Obj-C.
    void updateScrollerStyle() final;
    bool scrollbarPaintTimerIsActive() const;
    void startScrollbarPaintTimer();
    void stopScrollbarPaintTimer();
    void setVisibleScrollerThumbRect(const IntRect&);

    void scrollbarWidthChanged(WebCore::ScrollbarWidth) final;
private:

    void updateScrollerImps();

    // sendContentAreaScrolledSoon() will do the same work that sendContentAreaScrolled() does except
    // it does it after a zero-delay timer fires. This will prevent us from updating overlay scrollbar
    // information during layout.
    void sendContentAreaScrolled(const FloatSize& scrollDelta);
    void sendContentAreaScrolledSoon(const FloatSize& scrollDelta);

    void initialScrollbarPaintTimerFired();
    void sendContentAreaScrolledTimerFired();

    WheelEventTestMonitor* wheelEventTestMonitor() const;

    Timer m_initialScrollbarPaintTimer;
    Timer m_sendContentAreaScrolledTimer;

    RetainPtr<ScrollerImpPair> m_scrollerImpPair;
    RetainPtr<WebScrollerImpPairDelegate> m_scrollerImpPairDelegate;
    RetainPtr<WebScrollerImpDelegate> m_horizontalScrollerImpDelegate;
    RetainPtr<WebScrollerImpDelegate> m_verticalScrollerImpDelegate;

    FloatSize m_contentAreaScrolledTimerScrollDelta;
    IntRect m_visibleScrollerThumbRect;

    bool m_needsScrollerStyleUpdate { false };
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::ScrollbarsControllerMac)
    static bool isType(const WebCore::ScrollbarsController& controller) { return controller.isScrollbarsControllerMac(); }
SPECIALIZE_TYPE_TRAITS_END()

#endif // PLATFORM(MAC)
