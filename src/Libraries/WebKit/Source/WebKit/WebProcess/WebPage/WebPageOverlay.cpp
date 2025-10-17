/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 19, 2025.
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
#include "WebPageOverlay.h"

#include "WebFrame.h"
#include "WebPage.h"
#include <WebCore/GraphicsLayer.h>
#include <WebCore/LocalFrame.h>
#include <WebCore/PageOverlay.h>
#include <wtf/CheckedPtr.h>
#include <wtf/NeverDestroyed.h>

namespace WebKit {
using namespace WebCore;

static HashMap<WeakRef<PageOverlay>, WeakRef<WebPageOverlay>>& overlayMap()
{
    static NeverDestroyed<HashMap<WeakRef<PageOverlay>, WeakRef<WebPageOverlay>>> map;
    return map;
}

Ref<WebPageOverlay> WebPageOverlay::create(std::unique_ptr<WebPageOverlay::Client> client, PageOverlay::OverlayType overlayType)
{
    return adoptRef(*new WebPageOverlay(WTFMove(client), overlayType));
}

WebPageOverlay::WebPageOverlay(std::unique_ptr<WebPageOverlay::Client> client, PageOverlay::OverlayType overlayType)
    : m_overlay(PageOverlay::create(*this, overlayType))
    , m_client(WTFMove(client))
{
    ASSERT(m_client);
    overlayMap().add(*m_overlay, *this);
}

WebPageOverlay::~WebPageOverlay()
{
    if (!m_overlay)
        return;

    overlayMap().remove(*m_overlay);
    m_overlay = nullptr;
}

WebPageOverlay* WebPageOverlay::fromCoreOverlay(PageOverlay& overlay)
{
    return overlayMap().get(overlay);
}

void WebPageOverlay::setNeedsDisplay(const IntRect& dirtyRect)
{
    m_overlay->setNeedsDisplay(dirtyRect);
}

void WebPageOverlay::setNeedsDisplay()
{
    m_overlay->setNeedsDisplay();
}

void WebPageOverlay::clear()
{
    m_overlay->clear();
}

void WebPageOverlay::willMoveToPage(PageOverlay&, Page* page)
{
    RefPtr webPage = page ? WebPage::fromCorePage(*page) : nullptr;
    m_client->willMoveToPage(*this, webPage.get());
}

void WebPageOverlay::didMoveToPage(PageOverlay&, Page* page)
{
    RefPtr webPage = page ? WebPage::fromCorePage(*page) : nullptr;
    m_client->didMoveToPage(*this, webPage.get());
}

void WebPageOverlay::drawRect(PageOverlay&, GraphicsContext& context, const IntRect& dirtyRect)
{
    m_client->drawRect(*this, context, dirtyRect);
}

bool WebPageOverlay::mouseEvent(PageOverlay&, const PlatformMouseEvent& event)
{
    return m_client->mouseEvent(*this, event);
}

void WebPageOverlay::didScrollFrame(PageOverlay&, LocalFrame& frame)
{
    RefPtr webFrame = WebFrame::fromCoreFrame(frame);
    m_client->didScrollFrame(*this, webFrame.get());
}

#if PLATFORM(MAC)
auto WebPageOverlay::actionContextForResultAtPoint(FloatPoint location) -> std::optional<ActionContext>
{
    return m_client->actionContextForResultAtPoint(*this, location);
}

void WebPageOverlay::dataDetectorsDidPresentUI()
{
    m_client->dataDetectorsDidPresentUI(*this);
}

void WebPageOverlay::dataDetectorsDidChangeUI()
{
    m_client->dataDetectorsDidChangeUI(*this);
}

void WebPageOverlay::dataDetectorsDidHideUI()
{
    m_client->dataDetectorsDidHideUI(*this);
}
#endif // PLATFORM(MAC)

bool WebPageOverlay::copyAccessibilityAttributeStringValueForPoint(PageOverlay&, String attribute, FloatPoint parameter, String& value)
{
    return m_client->copyAccessibilityAttributeStringValueForPoint(*this, attribute, parameter, value);
}

bool WebPageOverlay::copyAccessibilityAttributeBoolValueForPoint(PageOverlay&, String attribute, FloatPoint parameter, bool& value)
{
    return m_client->copyAccessibilityAttributeBoolValueForPoint(*this, attribute, parameter, value);
}

Vector<String> WebPageOverlay::copyAccessibilityAttributeNames(PageOverlay&, bool parameterizedNames)
{
    return m_client->copyAccessibilityAttributeNames(*this, parameterizedNames);
}

} // namespace WebKit
