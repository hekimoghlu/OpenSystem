/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 8, 2024.
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

#if USE(CAIRO) || PLATFORM(GTK)

#include "UpdateInfo.h"
#include <WebCore/IntRect.h>
#include <WebCore/ShareableBitmap.h>
#include <cairo.h>

#if USE(CAIRO)
#include <WebCore/CairoUtilities.h>
#include <WebCore/GraphicsContextCairo.h>
#endif

#if USE(SKIA)
#include <WebCore/GraphicsContextSkia.h>
WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_BEGIN
#include <skia/core/SkCanvas.h>
#include <skia/core/SkSurface.h>
WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_END
#endif

namespace WebKit {
using namespace WebCore;

static const Seconds s_scrollHysteresisDuration { 300_ms };

static RefPtr<cairo_surface_t> createCairoImageSurfaceWithFastMalloc(const IntSize& size, double deviceScaleFactor)
{
    ASSERT(!size.isEmpty());
    IntSize scaledSize(size);
    scaledSize.scale(deviceScaleFactor);
    static cairo_user_data_key_t s_surfaceDataKey;
    int stride = cairo_format_stride_for_width(CAIRO_FORMAT_ARGB32, scaledSize.width());
    auto* surfaceData = fastZeroedMalloc(scaledSize.height() * stride);
    RefPtr<cairo_surface_t> surface = adoptRef(cairo_image_surface_create_for_data(static_cast<unsigned char*>(surfaceData), CAIRO_FORMAT_ARGB32, scaledSize.width(), scaledSize.height(), stride));
    cairo_surface_set_user_data(surface.get(), &s_surfaceDataKey, surfaceData, [](void* data) { fastFree(data); });
    cairo_surface_set_device_scale(surface.get(), deviceScaleFactor, deviceScaleFactor);
    return surface;
}

BackingStore::BackingStore(const IntSize& size, float deviceScaleFactor)
    : m_size(size)
    , m_deviceScaleFactor(deviceScaleFactor)
    , m_surface(createCairoImageSurfaceWithFastMalloc(m_size, m_deviceScaleFactor))
    , m_scrolledHysteresis([this](PAL::HysteresisState state) { if (state == PAL::HysteresisState::Stopped) m_scrollSurface = nullptr; }, s_scrollHysteresisDuration)
{
}

BackingStore::~BackingStore()
{
}

void BackingStore::paint(PlatformPaintContextPtr cr, const IntRect& rect)
{
    cairo_save(cr);
    cairo_set_operator(cr, CAIRO_OPERATOR_SOURCE);
    cairo_set_source_surface(cr, m_surface.get(), 0, 0);
    cairo_rectangle(cr, rect.x(), rect.y(), rect.width(), rect.height());
    cairo_fill(cr);
    cairo_restore(cr);
}

void BackingStore::incorporateUpdate(UpdateInfo&& updateInfo)
{
    ASSERT(m_size == updateInfo.viewSize);
    if (!updateInfo.bitmapHandle)
        return;

    auto bitmap = ShareableBitmap::create(WTFMove(*updateInfo.bitmapHandle));
    if (!bitmap)
        return;

#if ASSERT_ENABLED
    IntSize updateSize = expandedIntSize(updateInfo.updateRectBounds.size() * m_deviceScaleFactor);
    ASSERT(bitmap->size() == updateSize);
#endif

    scroll(updateInfo.scrollRect, updateInfo.scrollOffset);

    // Paint all update rects.
    IntPoint updateRectLocation = updateInfo.updateRectBounds.location();

#if USE(CAIRO)
    GraphicsContextCairo graphicsContext(m_surface.get());

    // When m_webPageProxy.drawsBackground() is false, bitmap contains transparent parts as a background of the webpage.
    // For such case, bitmap must be drawn using CompositeOperator::Copy to overwrite the existing surface.
    graphicsContext.setCompositeOperation(WebCore::CompositeOperator::Copy);

    for (const auto& updateRect : updateInfo.updateRects) {
        IntRect srcRect = updateRect;
        srcRect.move(-updateRectLocation.x(), -updateRectLocation.y());
        bitmap->paint(graphicsContext, m_deviceScaleFactor, updateRect.location(), srcRect);
    }
#elif USE(SKIA)
    cairo_surface_flush(m_surface.get());
    auto imageInfo = SkImageInfo::MakeN32Premul(cairo_image_surface_get_width(m_surface.get()), cairo_image_surface_get_height(m_surface.get()) , SkColorSpace::MakeSRGB());
    auto surface = SkSurfaces::WrapPixels(imageInfo, cairo_image_surface_get_data(m_surface.get()), cairo_image_surface_get_stride(m_surface.get()), nullptr);
    SkCanvas* canvas = surface ? surface->getCanvas() : nullptr;
    if (!canvas)
        return;

    GraphicsContextSkia graphicsContext(*canvas, RenderingMode::Unaccelerated, RenderingPurpose::ShareableLocalSnapshot);
    graphicsContext.setCompositeOperation(WebCore::CompositeOperator::Copy);
    for (const auto& updateRect : updateInfo.updateRects) {
        IntRect srcRect(updateRect);
        srcRect.move(-updateRectLocation.x(), -updateRectLocation.y());
        bitmap->paint(graphicsContext, m_deviceScaleFactor, updateRect.location(), srcRect);

        IntRect damage(updateRect.location(), srcRect.size());
        damage.scale(m_deviceScaleFactor);
        cairo_surface_mark_dirty_rectangle(m_surface.get(), damage.x(), damage.y(), damage.width(), damage.height());
    }
#endif
}

void BackingStore::scroll(const IntRect& scrollRect, const IntSize& scrollOffset)
{
    if (scrollOffset.isZero())
        return;

    IntRect targetRect = scrollRect;
    targetRect.move(scrollOffset);
    targetRect.intersect(scrollRect);
    if (targetRect.isEmpty())
        return;

    if (!m_scrollSurface)
        m_scrollSurface = createCairoImageSurfaceWithFastMalloc(m_size, m_deviceScaleFactor);

#if USE(SKIA)
    auto copyRectFromOneSurfaceToAnother = [](cairo_surface_t* src, cairo_surface_t* dst, const IntSize& sourceOffset, const IntRect& rect) {
        RefPtr<cairo_t> cr = adoptRef(cairo_create(dst));
        cairo_set_operator(cr.get(), CAIRO_OPERATOR_SOURCE);
        cairo_set_source_surface(cr.get(), src, sourceOffset.width(), sourceOffset.height());
        cairo_rectangle(cr.get(), rect.x(), rect.y(), rect.width(), rect.height());
        cairo_fill(cr.get());
    };
#endif

    copyRectFromOneSurfaceToAnother(m_surface.get(), m_scrollSurface.get(), scrollOffset, targetRect);
    copyRectFromOneSurfaceToAnother(m_scrollSurface.get(), m_surface.get(), { }, targetRect);
    m_scrolledHysteresis.impulse();
}

} // namespace WebKit

#endif // USE(CAIRO) || PLATFORM(GTK)
