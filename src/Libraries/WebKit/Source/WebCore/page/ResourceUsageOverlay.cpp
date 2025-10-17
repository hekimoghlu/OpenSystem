/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 27, 2023.
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

#if ENABLE(RESOURCE_USAGE)

#include "ResourceUsageOverlay.h"

#include "LocalFrame.h"
#include "LocalFrameView.h"
#include "Page.h"
#include "PageOverlayController.h"
#include "PlatformMouseEvent.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ResourceUsageOverlay);

Ref<ResourceUsageOverlay> ResourceUsageOverlay::create(Page& page)
{
    return adoptRef(*new ResourceUsageOverlay(page));
}

ResourceUsageOverlay::ResourceUsageOverlay(Page& page)
    : m_page(page)
    , m_overlay(PageOverlay::create(*this, PageOverlay::OverlayType::View))
{
    ASSERT(isMainThread());
    // Let the event loop cycle before continuing with initialization.
    // This way we'll have access to the FrameView's dimensions.
    callOnMainThread([weakThis = WeakPtr { *this }] {
        if (RefPtr protectedThis = weakThis.get())
            protectedThis->initialize();
    });
}

ResourceUsageOverlay::~ResourceUsageOverlay()
{
    ASSERT(isMainThread());
    platformDestroy();
    if (RefPtr page = m_page.get())
        page->pageOverlayController().uninstallPageOverlay(*m_overlay.copyRef(), PageOverlay::FadeMode::DoNotFade);
}

void ResourceUsageOverlay::initialize()
{
    RefPtr page = m_page.get();
    if (!page)
        return;
    auto* frameView = page->mainFrame().virtualView();
    if (!frameView)
        return;
    IntRect initialRect(frameView->width() / 2 - normalWidth / 2, frameView->height() - normalHeight - 20, normalWidth, normalHeight);

#if PLATFORM(IOS_FAMILY)
    // FIXME: The overlay should be stuck to the viewport instead of moving along with the page.
    initialRect.setY(20);
#endif

    RefPtr overlay = m_overlay;
    overlay->setFrame(initialRect);
    page->pageOverlayController().installPageOverlay(*overlay, PageOverlay::FadeMode::DoNotFade);
    platformInitialize();
}

bool ResourceUsageOverlay::mouseEvent(PageOverlay&, const PlatformMouseEvent& event)
{
    if (event.button() != MouseButton::Left)
        return false;

    RefPtr overlay = m_overlay;
    switch (event.type()) {
    case PlatformEvent::Type::MousePressed: {
        overlay->setShouldIgnoreMouseEventsOutsideBounds(false);
        m_dragging = true;
        IntPoint location = overlay->frame().location();
        m_dragPoint = event.position() + IntPoint(-location.x(), -location.y());
        return true;
    }
    case PlatformEvent::Type::MouseReleased:
        if (m_dragging) {
            overlay->setShouldIgnoreMouseEventsOutsideBounds(true);
            m_dragging = false;
            return true;
        }
        break;
    case PlatformEvent::Type::MouseMoved:
        if (m_dragging) {
            RefPtr page = m_page.get();
            if (!page)
                return false;
            IntRect newFrame = overlay->frame();

            // Move the new frame relative to the point where the drag was initiated.
            newFrame.setLocation(event.position());
            newFrame.moveBy(IntPoint(-m_dragPoint.x(), -m_dragPoint.y()));

            // Force the frame to stay inside the viewport entirely.
            if (newFrame.x() < 0)
                newFrame.setX(0);
            if (newFrame.y() < page->topContentInset())
                newFrame.setY(page->topContentInset());
            auto& frameView = *page->mainFrame().virtualView();
            if (newFrame.maxX() > frameView.width())
                newFrame.setX(frameView.width() - newFrame.width());
            if (newFrame.maxY() > frameView.height())
                newFrame.setY(frameView.height() - newFrame.height());

            overlay->setFrame(newFrame);
            overlay->setNeedsDisplay();
            return true;
        }
        break;
    default:
        break;
    }
    return false;
}

}

#endif
