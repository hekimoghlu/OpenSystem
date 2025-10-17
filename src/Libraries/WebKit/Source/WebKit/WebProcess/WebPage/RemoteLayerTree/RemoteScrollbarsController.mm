/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 9, 2025.
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
#include "RemoteScrollbarsController.h"

#if PLATFORM(MAC)

#include <WebCore/ScrollableArea.h>
#include <WebCore/ScrollbarThemeMac.h>
#include <WebCore/ScrollingCoordinator.h>
#include <pal/spi/mac/NSScrollerImpSPI.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteScrollbarsController);

RemoteScrollbarsController::RemoteScrollbarsController(WebCore::ScrollableArea& scrollableArea, WebCore::ScrollingCoordinator* coordinator)
    : ScrollbarsController(scrollableArea)
    , m_coordinator(ThreadSafeWeakPtr<WebCore::ScrollingCoordinator>(coordinator))
{
    if (auto scrollingCoordinator = m_coordinator.get())
        scrollingCoordinator->setScrollbarWidth(scrollableArea, scrollableArea.scrollbarWidthStyle());
}

void RemoteScrollbarsController::scrollbarLayoutDirectionChanged(WebCore::UserInterfaceLayoutDirection scrollbarLayoutDirection)
{
    if (RefPtr scrollingCoordinator = m_coordinator.get())
        scrollingCoordinator->setScrollbarLayoutDirection(scrollableArea(), scrollbarLayoutDirection);
}

void RemoteScrollbarsController::mouseEnteredContentArea()
{
    if (auto scrollingCoordinator = m_coordinator.get())
        scrollingCoordinator->setMouseIsOverContentArea(scrollableArea(), true);
}

void RemoteScrollbarsController::mouseExitedContentArea()
{
    if (auto scrollingCoordinator = m_coordinator.get())
        scrollingCoordinator->setMouseIsOverContentArea(scrollableArea(), false);
}

void RemoteScrollbarsController::mouseMovedInContentArea()
{
    if (auto scrollingCoordinator = m_coordinator.get())
        scrollingCoordinator->setMouseMovedInContentArea(scrollableArea());
}

void RemoteScrollbarsController::mouseEnteredScrollbar(WebCore::Scrollbar* scrollbar) const
{
    if (auto scrollingCoordinator = m_coordinator.get())
        scrollingCoordinator->setMouseIsOverScrollbar(scrollbar, true);
}

void RemoteScrollbarsController::mouseExitedScrollbar(WebCore::Scrollbar* scrollbar) const
{
    if (auto scrollingCoordinator = m_coordinator.get())
        scrollingCoordinator->setMouseIsOverScrollbar(scrollbar, false);
}

bool RemoteScrollbarsController::shouldScrollbarParticipateInHitTesting(WebCore::Scrollbar* scrollbar)
{
    // Non-overlay scrollbars should always participate in hit testing.
    ASSERT(scrollbar->isOverlayScrollbar());

    // Overlay scrollbars should participate in hit testing whenever they are at all visible.
    return scrollbar->orientation() == WebCore::ScrollbarOrientation::Horizontal ? m_horizontalOverlayScrollbarIsVisible :  m_verticalOverlayScrollbarIsVisible;
}

void RemoteScrollbarsController::setScrollbarVisibilityState(WebCore::ScrollbarOrientation orientation, bool isVisible)
{
    if (orientation == WebCore::ScrollbarOrientation::Horizontal)
        m_horizontalOverlayScrollbarIsVisible = isVisible;
    else
        m_verticalOverlayScrollbarIsVisible = isVisible;
}

bool RemoteScrollbarsController::shouldDrawIntoScrollbarLayer(WebCore::Scrollbar& scrollbar) const
{
    // For UI-side compositing we only draw scrollbars in the web process
    // for custom scrollbars
    return scrollbar.isCustomScrollbar() || scrollbar.isMockScrollbar();
}

bool RemoteScrollbarsController::shouldRegisterScrollbars() const
{
    return !scrollableArea().usesAsyncScrolling();
}

void RemoteScrollbarsController::setScrollbarMinimumThumbLength(WebCore::ScrollbarOrientation orientation, int minimumThumbLength)
{
    if (orientation == WebCore::ScrollbarOrientation::Horizontal)
        m_horizontalMinimumThumbLength = minimumThumbLength;
    else
        m_verticalMinimumThumbLength = minimumThumbLength;
}

int RemoteScrollbarsController::minimumThumbLength(WebCore::ScrollbarOrientation orientation)
{
    return orientation == WebCore::ScrollbarOrientation::Horizontal ? m_horizontalMinimumThumbLength : m_verticalMinimumThumbLength;
}

void RemoteScrollbarsController::updateScrollbarEnabledState(WebCore::Scrollbar& scrollbar)
{
    if (auto scrollingCoordinator = m_coordinator.get())
        scrollingCoordinator->setScrollbarEnabled(scrollbar);
}

void RemoteScrollbarsController::scrollbarWidthChanged(WebCore::ScrollbarWidth width)
{
    if (auto scrollingCoordinator = m_coordinator.get())
        scrollingCoordinator->setScrollbarWidth(scrollableArea(), width);

    updateScrollbarsThickness();
}

void RemoteScrollbarsController::updateScrollbarStyle()
{
    auto& theme = WebCore::ScrollbarTheme::theme();
    if (theme.isMockTheme())
        return;

    // The different scrollbar styles have different thicknesses, so we must re-set the
    // frameRect to the new thickness, and the re-layout below will ensure the position
    // and length are properly updated.
    updateScrollbarsThickness();

    scrollableArea().scrollbarStyleChanged(theme.usesOverlayScrollbars() ? WebCore::ScrollbarStyle::Overlay : WebCore::ScrollbarStyle::AlwaysVisible, true);
}

}
#endif // PLATFORM(MAC)
