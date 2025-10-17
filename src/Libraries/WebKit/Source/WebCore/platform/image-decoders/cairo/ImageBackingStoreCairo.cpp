/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 2, 2025.
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
#include "CairoUtilities.h"
#include "ImageBackingStore.h"

#include <cairo.h>

namespace WebCore {

PlatformImagePtr ImageBackingStore::image() const
{
    m_pixels->ref();
    RefPtr<cairo_surface_t> surface = adoptRef(cairo_image_surface_create_for_data(
        reinterpret_cast<unsigned char*>(const_cast<uint32_t*>(m_pixelsSpan.data())),
        CAIRO_FORMAT_ARGB32, size().width(), size().height(), size().width() * sizeof(uint32_t)));
    static cairo_user_data_key_t s_surfaceDataKey;
    cairo_surface_set_user_data(surface.get(), &s_surfaceDataKey, m_pixels.get(), [](void* data) {
        static_cast<DataSegment*>(data)->deref();
    });

    attachSurfaceUniqueID(surface.get());
    return surface;
}

} // namespace WebCore
