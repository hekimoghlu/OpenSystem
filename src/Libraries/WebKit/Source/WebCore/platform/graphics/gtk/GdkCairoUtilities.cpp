/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 23, 2025.
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
#include "GdkCairoUtilities.h"

#if USE(CAIRO)

#include "CairoUtilities.h"
#include "IntSize.h"
#include <cairo.h>
#include <gtk/gtk.h>
#include <mutex>
#include <wtf/NeverDestroyed.h>
#include <wtf/glib/GUniquePtr.h>

namespace WebCore {

GRefPtr<GdkPixbuf> cairoSurfaceToGdkPixbuf(cairo_surface_t* surface)
{
    IntSize size = cairoSurfaceSize(surface);
    return adoptGRef(gdk_pixbuf_get_from_surface(surface, 0, 0, size.width(), size.height()));
}

#if USE(GTK4)
GRefPtr<GdkTexture> cairoSurfaceToGdkTexture(cairo_surface_t* surface)
{
    ASSERT(cairo_image_surface_get_format(surface) == CAIRO_FORMAT_ARGB32);
    auto width = cairo_image_surface_get_width(surface);
    auto height = cairo_image_surface_get_height(surface);
    if (width <= 0 || height <= 0)
        return nullptr;
    auto stride = cairo_image_surface_get_stride(surface);
    auto* data = cairo_image_surface_get_data(surface);
    GRefPtr<GBytes> bytes = adoptGRef(g_bytes_new_with_free_func(data, height * stride, [](gpointer data) {
        cairo_surface_destroy(static_cast<cairo_surface_t*>(data));
    }, cairo_surface_reference(surface)));
    return adoptGRef(gdk_memory_texture_new(width, height, GDK_MEMORY_DEFAULT, bytes.get(), stride));
}
#endif

} // namespace WebCore

#endif // #if USE(CAIRO)
