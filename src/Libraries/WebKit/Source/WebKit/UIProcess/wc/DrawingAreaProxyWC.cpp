/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 16, 2024.
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
#include "DrawingAreaProxyWC.h"

#if USE(GRAPHICS_LAYER_WC)

#include "DrawingAreaMessages.h"
#include "MessageSenderInlines.h"
#include "UpdateInfo.h"
#include "WebPageProxy.h"
#include <WebCore/Region.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(DrawingAreaProxyWC);

Ref<DrawingAreaProxyWC> DrawingAreaProxyWC::create(WebPageProxy& page, WebProcessProxy& webProcessProxy)
{
    return adoptRef(*new DrawingAreaProxyWC(page, webProcessProxy));
}

DrawingAreaProxyWC::DrawingAreaProxyWC(WebPageProxy& webPageProxy, WebProcessProxy& webProcessProxy)
    : DrawingAreaProxy(DrawingAreaType::WC, webPageProxy, webProcessProxy)
{
}

void DrawingAreaProxyWC::paint(PlatformPaintContextPtr context, const WebCore::IntRect& rect, WebCore::Region& unpaintedRegion)
{
    unpaintedRegion = rect;

    if (!m_backingStore)
        return;
    m_backingStore->paint(context, rect);
    unpaintedRegion.subtract(WebCore::IntRect({ }, m_backingStore->size()));
}

void DrawingAreaProxyWC::deviceScaleFactorDidChange(CompletionHandler<void()>&& completionHandler)
{
    sizeDidChange();
    completionHandler();
}

void DrawingAreaProxyWC::sizeDidChange()
{
    discardBackingStore();
    m_currentBackingStoreStateID++;
    if (m_webPageProxy)
        send(Messages::DrawingArea::UpdateGeometryWC(m_currentBackingStoreStateID, m_size, m_webPageProxy->deviceScaleFactor(), m_webPageProxy->intrinsicDeviceScaleFactor()));
}

void DrawingAreaProxyWC::update(uint64_t backingStoreStateID, UpdateInfo&& updateInfo)
{
    if (backingStoreStateID == m_currentBackingStoreStateID)
        incorporateUpdate(WTFMove(updateInfo));
    send(Messages::DrawingArea::DisplayDidRefresh());
}

void DrawingAreaProxyWC::enterAcceleratedCompositingMode(uint64_t backingStoreStateID, const LayerTreeContext&)
{
    discardBackingStore();
}

void DrawingAreaProxyWC::incorporateUpdate(UpdateInfo&& updateInfo)
{
    if (updateInfo.updateRectBounds.isEmpty())
        return;

    if (!m_backingStore)
        m_backingStore.emplace(updateInfo.viewSize, updateInfo.deviceScaleFactor);

    RefPtr page = m_webPageProxy.get();
    if (!page)
        return;

    WebCore::Region damageRegion;
    if (updateInfo.scrollRect.isEmpty()) {
        for (const auto& rect : updateInfo.updateRects)
            damageRegion.unite(rect);
    } else
        damageRegion = WebCore::IntRect({ }, page->viewSize());

    m_backingStore->incorporateUpdate(WTFMove(updateInfo));

    page->setViewNeedsDisplay(damageRegion);
}

void DrawingAreaProxyWC::discardBackingStore()
{
    m_backingStore = std::nullopt;
}

} // namespace WebKit

#endif // USE(GRAPHICS_LAYER_WC)
