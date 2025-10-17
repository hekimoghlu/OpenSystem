/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 6, 2023.
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
#include "WKImageCairo.h"

#if USE(CAIRO)
#include "WKSharedAPICast.h"
#include "WebImage.h"
#include <WebCore/GraphicsContextCairo.h>
#include <WebCore/ShareableBitmap.h>
#include <cairo.h>

cairo_surface_t* WKImageCreateCairoSurface(WKImageRef imageRef)
{
    // We cannot pass a RefPtr through the API here, so we just leak the reference.
    return WebKit::toImpl(imageRef)->createCairoSurface().leakRef();
}

WKImageRef WKImageCreateFromCairoSurface(cairo_surface_t* surface, WKImageOptions options)
{
    WebCore::IntSize imageSize(cairo_image_surface_get_width(surface), cairo_image_surface_get_height(surface));
    auto webImage = WebKit::WebImage::create(imageSize, WebKit::toImageOptions(options), WebCore::DestinationColorSpace::SRGB());
    if (!webImage->context())
        return nullptr;
    auto& graphicsContext = *webImage->context();

    cairo_t* cr = graphicsContext.platformContext()->cr();
    cairo_set_source_surface(cr, surface, 0, 0);
    cairo_set_operator(cr, CAIRO_OPERATOR_SOURCE);
    cairo_rectangle(cr, 0, 0, imageSize.width(), imageSize.height());
    cairo_fill(cr);

    return toAPI(webImage.leakRef());
}

#endif // USE(CAIRO)
