/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 5, 2025.
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
#include "ImageAdapter.h"

#include "BitmapImage.h"
#include "GdkCairoUtilities.h"
#include "GdkSkiaUtilities.h"
#include "SharedBuffer.h"
#include <cairo.h>
#include <gdk/gdk.h>
#include <wtf/glib/GSpanExtras.h>
#include <wtf/glib/GUniquePtr.h>

namespace WebCore {

static Ref<Image> loadImageFromGResource(const char* iconName)
{
    auto icon = BitmapImage::create();
    GUniquePtr<char> path(g_strdup_printf("/org/webkitgtk/resources/images/%s", iconName));
    GRefPtr<GBytes> data = adoptGRef(g_resources_lookup_data(path.get(), G_RESOURCE_LOOKUP_FLAGS_NONE, nullptr));
    ASSERT(data);
    icon->setData(SharedBuffer::create(span(data)), true);
    return icon;
}

Ref<Image> ImageAdapter::loadPlatformResource(const char* name)
{
    return loadImageFromGResource(name);
}

void ImageAdapter::invalidate()
{
}

GRefPtr<GdkPixbuf> ImageAdapter::gdkPixbuf()
{
    RefPtr nativeImage = image().currentNativeImage();
    if (!nativeImage)
        return nullptr;

    auto& surface = nativeImage->platformImage();
#if USE(CAIRO)
    return cairoSurfaceToGdkPixbuf(surface.get());
#elif USE(SKIA)
    return skiaImageToGdkPixbuf(*surface.get());
#endif
}

#if USE(GTK4)
GRefPtr<GdkTexture> ImageAdapter::gdkTexture()
{
    RefPtr nativeImage = image().currentNativeImage();
    if (!nativeImage)
        return nullptr;

    auto& surface = nativeImage->platformImage();
#if USE(CAIRO)
    return cairoSurfaceToGdkTexture(surface.get());
#elif USE(SKIA)
    return skiaImageToGdkTexture(*surface.get());
#endif
}
#endif

} // namespace WebCore
