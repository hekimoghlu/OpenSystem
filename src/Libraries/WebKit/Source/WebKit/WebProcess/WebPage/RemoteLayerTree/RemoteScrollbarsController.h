/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 29, 2024.
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

#include <WebCore/NSScrollerImpDetails.h>
#include <WebCore/ScrollbarsController.h>
#include <WebCore/UserInterfaceLayoutDirection.h>
#include <wtf/RetainPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/ThreadSafeWeakPtr.h>

namespace WebCore {
class ScrollingCoordinator;
}

namespace WebKit {

class RemoteScrollbarsController final : public WebCore::ScrollbarsController {
    WTF_MAKE_TZONE_ALLOCATED(RemoteScrollbarsController);
public:
    RemoteScrollbarsController(WebCore::ScrollableArea&, WebCore::ScrollingCoordinator*);
    ~RemoteScrollbarsController() = default;
    void mouseEnteredContentArea() final;
    void mouseExitedContentArea()  final;
    void mouseMovedInContentArea() final;
    void mouseEnteredScrollbar(WebCore::Scrollbar*) const final;
    void mouseExitedScrollbar(WebCore::Scrollbar*) const final;
    bool shouldScrollbarParticipateInHitTesting(WebCore::Scrollbar*) final;

    void setScrollbarMinimumThumbLength(WebCore::ScrollbarOrientation, int) final;
    void setScrollbarVisibilityState(WebCore::ScrollbarOrientation, bool) final;
    bool shouldDrawIntoScrollbarLayer(WebCore::Scrollbar&) const final;
    bool shouldRegisterScrollbars() const final;
    int minimumThumbLength(WebCore::ScrollbarOrientation) final;
    void updateScrollbarEnabledState(WebCore::Scrollbar&) final;
    void scrollbarLayoutDirectionChanged(WebCore::UserInterfaceLayoutDirection) final;

    void updateScrollbarStyle() final;

    void scrollbarWidthChanged(WebCore::ScrollbarWidth) final;

    bool isRemoteScrollbarsController() const final { return true; }

private:
    bool m_horizontalOverlayScrollbarIsVisible { false };
    bool m_verticalOverlayScrollbarIsVisible { false };

    int m_horizontalMinimumThumbLength { 0 };
    int m_verticalMinimumThumbLength { 0 };
    ThreadSafeWeakPtr<WebCore::ScrollingCoordinator> m_coordinator;
};

} // namespace WebKit

SPECIALIZE_TYPE_TRAITS_BEGIN(WebKit::RemoteScrollbarsController)
    static bool isType(const WebCore::ScrollbarsController& controller) { return controller.isRemoteScrollbarsController(); }
SPECIALIZE_TYPE_TRAITS_END()

#endif // PLATFORM(MAC)
