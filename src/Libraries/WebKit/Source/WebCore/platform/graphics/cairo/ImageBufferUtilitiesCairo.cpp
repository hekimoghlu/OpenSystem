/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 24, 2022.
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
#include "ImageBufferUtilitiesCairo.h"

#if USE(CAIRO)

#include "CairoUtilities.h"
#include "MIMETypeRegistry.h"
#include "RefPtrCairo.h"
#include <cairo.h>
#include <wtf/text/Base64.h>
#include <wtf/text/CString.h>
#include <wtf/text/WTFString.h>

#if PLATFORM(GTK)
#include "GRefPtrGtk.h"
#include "GdkCairoUtilities.h"
#include <gtk/gtk.h>
#include <wtf/glib/GUniquePtr.h>
#endif

namespace WebCore {

#if !PLATFORM(GTK)
static cairo_status_t writeFunction(void* output, const unsigned char* data, unsigned length)
{
    if (!reinterpret_cast<Vector<uint8_t>*>(output)->tryAppend(std::span { data, length }))
        return CAIRO_STATUS_WRITE_ERROR;
    return CAIRO_STATUS_SUCCESS;
}

static bool encodeImage(cairo_surface_t* image, const String& mimeType, Vector<uint8_t>* output)
{
    ASSERT_UNUSED(mimeType, mimeType == "image/png"_s); // Only PNG output is supported for now.

    return cairo_surface_write_to_png_stream(image, writeFunction, output) == CAIRO_STATUS_SUCCESS;
}

Vector<uint8_t> encodeData(cairo_surface_t* image, const String& mimeType, std::optional<double>)
{
    Vector<uint8_t> encodedImage;
    if (!image || !encodeImage(image, mimeType, &encodedImage))
        return { };
    return encodedImage;
}
#else
static bool encodeImage(cairo_surface_t* surface, const String& mimeType, std::optional<double> quality, GUniqueOutPtr<gchar>& buffer, gsize& bufferSize)
{
    // List of supported image encoding types comes from the GdkPixbuf documentation.
    // http://developer.gnome.org/gdk-pixbuf/stable/gdk-pixbuf-File-saving.html#gdk-pixbuf-save-to-bufferv

    String type = mimeType.substring(sizeof "image");
    if (type != "jpeg"_s && type != "png"_s && type != "tiff"_s && type != "ico"_s && type != "bmp"_s)
        return false;

    auto pixbuf = cairoSurfaceToGdkPixbuf(surface);
    if (!pixbuf)
        return false;

    GUniqueOutPtr<GError> error;
    if (type == "jpeg"_s && quality && *quality >= 0.0 && *quality <= 1.0) {
        String qualityString = String::number(static_cast<int>(*quality * 100.0 + 0.5));
        gdk_pixbuf_save_to_buffer(pixbuf.get(), &buffer.outPtr(), &bufferSize, type.utf8().data(), &error.outPtr(), "quality", qualityString.utf8().data(), NULL);
    } else
        gdk_pixbuf_save_to_buffer(pixbuf.get(), &buffer.outPtr(), &bufferSize, type.utf8().data(), &error.outPtr(), NULL);

    return !error;
}

Vector<uint8_t> encodeData(cairo_surface_t* image, const String& mimeType, std::optional<double> quality)
{
    GUniqueOutPtr<gchar> buffer;
    gsize bufferSize;
    if (!encodeImage(image, mimeType, quality, buffer, bufferSize))
        return { };

    return std::span { reinterpret_cast<const uint8_t*>(buffer.get()), bufferSize };
}
#endif // !PLATFORM(GTK)

} // namespace WebCore

#endif // USE(CAIRO)
