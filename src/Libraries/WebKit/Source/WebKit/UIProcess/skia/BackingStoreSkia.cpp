/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 23, 2022.
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
#include "BackingStore.h"

#if USE(SKIA)

#include "UpdateInfo.h"
#include <WebCore/GraphicsContextSkia.h>
#include <skia/core/SkBitmap.h>
#include <skia/core/SkCanvas.h>

namespace WebKit {

BackingStore::BackingStore(const WebCore::IntSize& size, float deviceScaleFactor)
    : m_size(size)
    , m_deviceScaleFactor(deviceScaleFactor)
{
    WebCore::IntSize scaledSize(size);
    scaledSize.scale(deviceScaleFactor);
    auto info = SkImageInfo::MakeN32Premul(scaledSize.width(), scaledSize.height(), SkColorSpace::MakeSRGB());
    m_surface = SkSurfaces::Raster(info);
    RELEASE_ASSERT(m_surface);
}

BackingStore::~BackingStore() = default;

void BackingStore::paint(PlatformPaintContextPtr cr, const WebCore::IntRect& rect)
{
    cr->save();
    cr->scale(1 / m_deviceScaleFactor, 1 / m_deviceScaleFactor);
    WebCore::IntRect scaledRect { rect };
    scaledRect.scale(m_deviceScaleFactor);
    cr->clipIRect(scaledRect);
    SkSamplingOptions option { SkFilterMode::kLinear };
    m_surface->draw(cr, 0, 0, option, nullptr);
    cr->restore();
}

void BackingStore::incorporateUpdate(UpdateInfo&& updateInfo)
{
    ASSERT(m_size == updateInfo.viewSize);
    if (!updateInfo.bitmapHandle)
        return;

    auto bitmap = WebCore::ShareableBitmap::create(WTFMove(*updateInfo.bitmapHandle));
    if (!bitmap)
        return;

#if ASSERT_ENABLED
    WebCore::IntSize updateSize = expandedIntSize(updateInfo.updateRectBounds.size() * m_deviceScaleFactor);
    ASSERT(bitmap->size() == updateSize);
#endif

    scroll(updateInfo.scrollRect, updateInfo.scrollOffset);

    // Paint all update rects.
    WebCore::IntPoint updateRectLocation = updateInfo.updateRectBounds.location();

    SkCanvas* canvas = m_surface->getCanvas();
    if (!canvas)
        return;

    WebCore::GraphicsContextSkia graphicsContext(*canvas, WebCore::RenderingMode::Unaccelerated, WebCore::RenderingPurpose::ShareableLocalSnapshot);
    graphicsContext.setCompositeOperation(WebCore::CompositeOperator::Copy);

    for (const auto& updateRect : updateInfo.updateRects) {
        WebCore::IntRect srcRect = updateRect;
        srcRect.move(-updateRectLocation.x(), -updateRectLocation.y());
        bitmap->paint(graphicsContext, m_deviceScaleFactor, updateRect.location(), srcRect);
    }
}

void BackingStore::scroll(const WebCore::IntRect& scrollRect, const WebCore::IntSize& scrollOffset)
{
    if (scrollOffset.isZero())
        return;

    WebCore::IntRect targetRect = scrollRect;
    targetRect.move(scrollOffset);
    targetRect.intersect(scrollRect);
    if (targetRect.isEmpty())
        return;

    WebCore::IntSize scaledScrollOffset = scrollOffset;
    targetRect.scale(m_deviceScaleFactor);
    scaledScrollOffset.scale(m_deviceScaleFactor, m_deviceScaleFactor);

    SkBitmap bitmap;
    bitmap.allocPixels(m_surface->imageInfo().makeWH(targetRect.width(), targetRect.height()));

    if (!m_surface->readPixels(bitmap, targetRect.x() - scaledScrollOffset.width(), targetRect.y() - scaledScrollOffset.height()))
        return;

    m_surface->writePixels(bitmap, targetRect.x(), targetRect.y());
}

} // namespace WebKit

#endif // USE(SKIA)
