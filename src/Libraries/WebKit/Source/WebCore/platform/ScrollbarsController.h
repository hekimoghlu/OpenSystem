/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 8, 2025.
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
#include "FloatSize.h"
#include "UserInterfaceLayoutDirection.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakHashSet.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class Scrollbar;
class ScrollableArea;
enum class ScrollbarOrientation : uint8_t;
enum class ScrollbarWidth : uint8_t;

class ScrollbarsController {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(ScrollbarsController, WEBCORE_EXPORT);
    WTF_MAKE_NONCOPYABLE(ScrollbarsController);
public:
    WEBCORE_EXPORT static std::unique_ptr<ScrollbarsController> create(ScrollableArea&);

    WEBCORE_EXPORT explicit ScrollbarsController(ScrollableArea&);
    virtual ~ScrollbarsController() = default;
    
    ScrollableArea& scrollableArea() const { return m_scrollableArea; }

    bool scrollbarAnimationsUnsuspendedByUserInteraction() const { return m_scrollbarAnimationsUnsuspendedByUserInteraction; }
    void setScrollbarAnimationsUnsuspendedByUserInteraction(bool unsuspended) { m_scrollbarAnimationsUnsuspendedByUserInteraction = unsuspended; }
    
    WEBCORE_EXPORT virtual bool isRemoteScrollbarsController() const { return false; }
    WEBCORE_EXPORT virtual bool isScrollbarsControllerMac() const { return false; }
    WEBCORE_EXPORT virtual bool isScrollbarsControllerMock() const { return false; }

    bool shouldSuspendScrollbarAnimations() const;

    virtual void notifyContentAreaScrolled(const FloatSize&) { }

    WEBCORE_EXPORT virtual void cancelAnimations();

    WEBCORE_EXPORT virtual void didBeginScrollGesture();
    WEBCORE_EXPORT virtual void didEndScrollGesture();
    WEBCORE_EXPORT virtual void mayBeginScrollGesture();

    WEBCORE_EXPORT virtual void contentAreaWillPaint() const { }
    WEBCORE_EXPORT virtual void mouseEnteredContentArea() { }
    WEBCORE_EXPORT virtual void mouseExitedContentArea() { }
    WEBCORE_EXPORT virtual void mouseMovedInContentArea() { }
    WEBCORE_EXPORT virtual void mouseEnteredScrollbar(Scrollbar*) const { }
    WEBCORE_EXPORT virtual void mouseExitedScrollbar(Scrollbar*) const { }
    WEBCORE_EXPORT virtual void mouseIsDownInScrollbar(Scrollbar*, bool) const { }
    WEBCORE_EXPORT virtual void willStartLiveResize() { }
    WEBCORE_EXPORT virtual void contentsSizeChanged() const { }
    WEBCORE_EXPORT virtual void willEndLiveResize() { }
    WEBCORE_EXPORT virtual void contentAreaDidShow() { }
    WEBCORE_EXPORT virtual void contentAreaDidHide() { }

    WEBCORE_EXPORT virtual void lockOverlayScrollbarStateToHidden(bool) { }
    WEBCORE_EXPORT virtual bool scrollbarsCanBeActive() const { return true; }

    WEBCORE_EXPORT virtual void didAddVerticalScrollbar(Scrollbar*) { }
    WEBCORE_EXPORT virtual void willRemoveVerticalScrollbar(Scrollbar*) { }
    WEBCORE_EXPORT virtual void didAddHorizontalScrollbar(Scrollbar*) { }
    WEBCORE_EXPORT virtual void willRemoveHorizontalScrollbar(Scrollbar*) { }

    WEBCORE_EXPORT virtual void invalidateScrollbarPartLayers(Scrollbar*) { }

    WEBCORE_EXPORT virtual void verticalScrollbarLayerDidChange() { }
    WEBCORE_EXPORT virtual void horizontalScrollbarLayerDidChange() { }

    WEBCORE_EXPORT virtual bool shouldScrollbarParticipateInHitTesting(Scrollbar*) { return true; }

    WEBCORE_EXPORT virtual String horizontalScrollbarStateForTesting() const { return emptyString(); }
    WEBCORE_EXPORT virtual String verticalScrollbarStateForTesting() const { return emptyString(); }

    WEBCORE_EXPORT virtual void setScrollbarVisibilityState(ScrollbarOrientation, bool) { }

    WEBCORE_EXPORT virtual bool shouldDrawIntoScrollbarLayer(Scrollbar&) const { return true; }
    WEBCORE_EXPORT virtual bool shouldRegisterScrollbars() const { return true; }
    WEBCORE_EXPORT virtual void updateScrollbarEnabledState(Scrollbar&) { }

    WEBCORE_EXPORT virtual void setScrollbarMinimumThumbLength(WebCore::ScrollbarOrientation, int) { }
    WEBCORE_EXPORT virtual int minimumThumbLength(WebCore::ScrollbarOrientation) { return 0; }
    WEBCORE_EXPORT virtual void scrollbarLayoutDirectionChanged(UserInterfaceLayoutDirection) { }

    WEBCORE_EXPORT virtual void updateScrollerStyle() { }

    WEBCORE_EXPORT void updateScrollbarsThickness();

    WEBCORE_EXPORT virtual void updateScrollbarStyle() { }

    WEBCORE_EXPORT virtual void scrollbarWidthChanged(WebCore::ScrollbarWidth) { }

private:
    ScrollableArea& m_scrollableArea;
    bool m_scrollbarAnimationsUnsuspendedByUserInteraction { true };
};

} // namespace WebCore
