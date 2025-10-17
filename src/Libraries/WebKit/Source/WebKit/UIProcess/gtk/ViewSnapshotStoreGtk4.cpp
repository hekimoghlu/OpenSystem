/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 6, 2022.
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

#if USE(GTK4)

namespace WebKit {
using namespace WebCore;

Ref<ViewSnapshot> ViewSnapshot::create(GRefPtr<GdkTexture>&& texture)
{
    return adoptRef(*new ViewSnapshot(WTFMove(texture)));
}

ViewSnapshot::ViewSnapshot(GRefPtr<GdkTexture>&& texture)
    : m_texture(WTFMove(texture))
{
    if (hasImage())
        ViewSnapshotStore::singleton().didAddImageToSnapshot(*this);
}

bool ViewSnapshot::hasImage() const
{
    return m_texture;
}

void ViewSnapshot::clearImage()
{
    if (!hasImage())
        return;

    ViewSnapshotStore::singleton().willRemoveImageFromSnapshot(*this);

    m_texture = nullptr;
}

size_t ViewSnapshot::estimatedImageSizeInBytes() const
{
    if (!m_texture)
        return 0;

    int width = gdk_texture_get_width(m_texture.get());
    int height = gdk_texture_get_height(m_texture.get());

    // Unfortunately we don't have a way to get size of a texture in
    // bytes, so we'll have to make something up. Let's assume that
    // pixel == 4 bytes.
    return width * height * 4;
}

WebCore::IntSize ViewSnapshot::size() const
{
    if (!m_texture)
        return { };

    int width = gdk_texture_get_width(m_texture.get());
    int height = gdk_texture_get_height(m_texture.get());

    return { width, height };
}

} // namespace WebKit

#endif
