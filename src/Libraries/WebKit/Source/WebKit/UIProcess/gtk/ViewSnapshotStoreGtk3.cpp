/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 16, 2025.
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
#include "ViewSnapshotStore.h"

#if !USE(GTK4)

#include <cairo.h>

namespace WebKit {
using namespace WebCore;

Ref<ViewSnapshot> ViewSnapshot::create(RefPtr<cairo_surface_t>&& surface)
{
    return adoptRef(*new ViewSnapshot(WTFMove(surface)));
}

ViewSnapshot::ViewSnapshot(RefPtr<cairo_surface_t>&& surface)
    : m_surface(WTFMove(surface))
{
    if (hasImage())
        ViewSnapshotStore::singleton().didAddImageToSnapshot(*this);
}

bool ViewSnapshot::hasImage() const
{
    return !!m_surface;
}

void ViewSnapshot::clearImage()
{
    if (!hasImage())
        return;

    ViewSnapshotStore::singleton().willRemoveImageFromSnapshot(*this);

    m_surface = nullptr;
}

size_t ViewSnapshot::estimatedImageSizeInBytes() const
{
    if (!m_surface)
        return 0;

    cairo_surface_t* surface = m_surface.get();
    int stride = cairo_image_surface_get_stride(surface);
    int height = cairo_image_surface_get_width(surface);

    return stride * height;
}

WebCore::IntSize ViewSnapshot::size() const
{
    if (!m_surface)
        return { };

    cairo_surface_t* surface = m_surface.get();
    int width = cairo_image_surface_get_width(surface);
    int height = cairo_image_surface_get_height(surface);

    return { width, height };
}

} // namespace WebKit

#endif
