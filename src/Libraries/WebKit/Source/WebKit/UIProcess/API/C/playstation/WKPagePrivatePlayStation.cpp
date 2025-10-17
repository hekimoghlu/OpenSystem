/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 14, 2024.
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
#include "WKPagePrivatePlayStation.h"

#include "DrawingAreaProxy.h"
#include "DrawingAreaProxyCoordinatedGraphics.h"
#include "NativeWebKeyboardEvent.h"
#include "NativeWebMouseEvent.h"
#include "NativeWebWheelEvent.h"
#include "WKAPICast.h"
#include "WebEventFactory.h"
#include "WebPageProxy.h"
#include <WebCore/Region.h>
#include <wpe/wpe.h>

#if USE(CAIRO)
#include <cairo.h>
#endif

#if USE(SKIA)
#include <skia/core/SkCanvas.h>
#endif

#if USE(GRAPHICS_LAYER_WC)
#include "DrawingAreaProxyWC.h"
#endif

static void drawPageBackground(WebKit::PlatformPaintContextPtr ctx, const std::optional<WebCore::Color>& backgroundColor, const WebCore::IntRect& rect)
{
    if (!backgroundColor || backgroundColor.value().isVisible())
        return;

#if USE(CAIRO)
    auto [r, g, b, a] = backgroundColor.value().toColorTypeLossy<WebCore::SRGBA<uint8_t>>().resolved();

    cairo_set_source_rgba(ctx, r, g, b, a);
    cairo_rectangle(ctx, rect.x(), rect.y(), rect.width(), rect.height());
    cairo_set_operator(ctx, CAIRO_OPERATOR_OVER);
    cairo_fill(ctx);
#elif USE(SKIA)
    ctx->clear(SkColor(backgroundColor.value()));
#endif
}

void WKPageHandleKeyboardEvent(WKPageRef pageRef, WKKeyboardEvent event)
{
    using WebKit::NativeWebKeyboardEvent;

    wpe_input_keyboard_event wpeEvent;
    wpeEvent.time = 0;
    wpeEvent.key_code = event.virtualKeyCode;
    wpeEvent.hardware_key_code = event.virtualKeyCode;
    wpeEvent.modifiers = 0; // TODO: Handle modifiers.
    switch (event.type) {
    case kWKEventKeyDown:
        wpeEvent.pressed = true;
        break;
    case kWKEventKeyUp:
        wpeEvent.pressed = false;
        break;
    default:
        ASSERT_NOT_REACHED();
        return;
    }

    NativeWebKeyboardEvent::HandledByInputMethod handledByInputMethod = NativeWebKeyboardEvent::HandledByInputMethod::No;
    std::optional<Vector<WebCore::CompositionUnderline>> preeditUnderlines;
    std::optional<WebKit::EditingRange> preeditSelectionRange;
    WebKit::toImpl(pageRef)->handleKeyboardEvent(NativeWebKeyboardEvent(&wpeEvent, ""_s, false, handledByInputMethod, WTFMove(preeditUnderlines), WTFMove(preeditSelectionRange)));
}

void WKPageHandleMouseEvent(WKPageRef pageRef, WKMouseEvent event)
{
    using WebKit::NativeWebMouseEvent;

    wpe_input_pointer_event wpeEvent;

    switch (event.type) {
    case kWKEventMouseDown:
        wpeEvent.type = wpe_input_pointer_event_type_button;
        wpeEvent.state = 1;
        break;
    case kWKEventMouseUp:
        wpeEvent.type = wpe_input_pointer_event_type_button;
        wpeEvent.state = 0;
        break;
    case kWKEventMouseMove:
        wpeEvent.type = wpe_input_pointer_event_type_motion;
        wpeEvent.state = 0;
        break;
    default:
        ASSERT_NOT_REACHED();
        return;
    }

    switch (event.button) {
    case kWKEventMouseButtonLeftButton:
        wpeEvent.button = 1;
        break;
    case kWKEventMouseButtonMiddleButton:
        wpeEvent.button = 3;
        break;
    case kWKEventMouseButtonRightButton:
        wpeEvent.button = 2;
        break;
    case kWKEventMouseButtonNoButton:
        wpeEvent.button = 0;
        break;
    default:
        ASSERT_NOT_REACHED();
        return;
    }
    wpeEvent.time = 0;
    wpeEvent.x = event.position.x;
    wpeEvent.y = event.position.y;
    wpeEvent.modifiers = 0; // TODO: Handle modifiers.

    const float deviceScaleFactor = 1;

    WebKit::toImpl(pageRef)->handleMouseEvent(NativeWebMouseEvent(&wpeEvent, deviceScaleFactor));
}

void WKPageHandleWheelEvent(WKPageRef pageRef, WKWheelEvent event)
{
    using WebKit::WebWheelEvent;
    using WebKit::NativeWebWheelEvent;

    const float deviceScaleFactor = 1;
    int positionX = event.position.x;
    int positionY = event.position.y;

    struct wpe_input_axis_event xEvent = {
        wpe_input_axis_event_type_motion,
        0, positionX, positionY,
        1, static_cast<int32_t>(event.delta.width), 0
    };

    WebKit::toImpl(pageRef)->handleNativeWheelEvent(NativeWebWheelEvent(&xEvent, deviceScaleFactor, WebWheelEvent::Phase::PhaseNone, WebWheelEvent::Phase::PhaseNone));

    struct wpe_input_axis_event yEvent = {
        wpe_input_axis_event_type_motion,
        0, positionX, positionY,
        0, static_cast<int32_t>(event.delta.height), 0
    };

    WebKit::toImpl(pageRef)->handleNativeWheelEvent(NativeWebWheelEvent(&yEvent, deviceScaleFactor, WebWheelEvent::Phase::PhaseNone, WebWheelEvent::Phase::PhaseNone));
}

void WKPagePaint(WKPageRef pageRef, unsigned char* surfaceData, WKSize wkSurfaceSize, WKRect wkPaintRect)
{
    auto surfaceSize = WebKit::toIntSize(wkSurfaceSize);
    auto paintRect = WebKit::toIntRect(wkPaintRect);
    if (!surfaceData || surfaceSize.isEmpty())
        return;

#if USE(CAIRO)
    const cairo_format_t format = CAIRO_FORMAT_ARGB32;
    cairo_surface_t* surface = cairo_image_surface_create_for_data(surfaceData, format, surfaceSize.width(), surfaceSize.height(), cairo_format_stride_for_width(format, surfaceSize.width()));
    if (cairo_surface_status(surface) != CAIRO_STATUS_SUCCESS)
        return;

    cairo_t* ctx = cairo_create(surface);
#elif USE(SKIA)
    auto info = SkImageInfo::MakeN32Premul(surfaceSize.width(), surfaceSize.height(), SkColorSpace::MakeSRGB());
    auto surface = SkSurfaces::WrapPixels(info, surfaceData, info.minRowBytes(), nullptr);
    auto ctx = surface->getCanvas();
#endif
    auto page = WebKit::toImpl(pageRef);
    auto& backgroundColor = page->backgroundColor();
    page->endPrinting();

    if (auto* drawingArea = page->drawingArea()) {
        // FIXME: We should port WebKit1's rect coalescing logic here.
        WebCore::Region unpaintedRegion;
#if USE(GRAPHICS_LAYER_WC)
        downcast<WebKit::DrawingAreaProxyWC>(drawingArea)->paint(ctx, paintRect, unpaintedRegion);
#else
        downcast<WebKit::DrawingAreaProxyCoordinatedGraphics>(drawingArea)->paint(ctx, paintRect, unpaintedRegion);
#endif

        for (const auto& rect : unpaintedRegion.rects())
            drawPageBackground(ctx, backgroundColor, rect);
    } else
        drawPageBackground(ctx, backgroundColor, paintRect);

#if USE(CAIRO)
    cairo_destroy(ctx);
    cairo_surface_destroy(surface);
#endif
}
