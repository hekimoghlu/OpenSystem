/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 26, 2022.
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
#include "GdkSkiaUtilities.h"

#if USE(SKIA)

#if !USE(GTK4)
#include <graphics/cairo/RefPtrCairo.h>
#endif

WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_BEGIN
#include <skia/core/SkPixmap.h>
WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_END

namespace WebCore {

#if USE(GTK4)
GRefPtr<GdkTexture> skiaImageToGdkTexture(SkImage& image)
{
    SkPixmap pixmap;
    if (!image.peekPixels(&pixmap))
        return { };

    GRefPtr<GBytes> bytes = adoptGRef(g_bytes_new_with_free_func(pixmap.addr(), pixmap.computeByteSize(), [](gpointer data) {
        static_cast<SkImage*>(data)->unref();
    }, SkRef(&image)));

    return adoptGRef(gdk_memory_texture_new(pixmap.width(), pixmap.height(), GDK_MEMORY_DEFAULT, bytes.get(), pixmap.rowBytes()));
}

#else

RefPtr<cairo_surface_t> skiaImageToCairoSurface(SkImage& image)
{
    SkPixmap pixmap;
    if (!image.peekPixels(&pixmap))
        return { };

    RefPtr<cairo_surface_t> surface = adoptRef(cairo_image_surface_create_for_data(pixmap.writable_addr8(0, 0), CAIRO_FORMAT_ARGB32, pixmap.width(), pixmap.height(), pixmap.rowBytes()));
    if (cairo_surface_status(surface.get()) != CAIRO_STATUS_SUCCESS)
        return { };

    static cairo_user_data_key_t surfaceDataKey;
    cairo_surface_set_user_data(surface.get(), &surfaceDataKey, SkRef(&image), [](void* data) {
        static_cast<SkImage*>(data)->unref();
    });

    return surface;
}
#endif

GRefPtr<GdkPixbuf> skiaImageToGdkPixbuf(SkImage& image)
{
#if USE(GTK4)
    auto texture = skiaImageToGdkTexture(image);
    if (!texture)
        return { };

    return adoptGRef(gdk_pixbuf_get_from_texture(texture.get()));
#else
    RefPtr surface = skiaImageToCairoSurface(image);
    if (!surface)
        return { };

    return adoptGRef(gdk_pixbuf_get_from_surface(surface.get(), 0, 0, cairo_image_surface_get_width(surface.get()), cairo_image_surface_get_height(surface.get())));
#endif
}

} // namespace WebCore

#endif // #if USE(SKIA)
