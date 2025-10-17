/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 11, 2024.
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
#include "RemoteResourceCache.h"

#if ENABLE(GPU_PROCESS)

#include "ArgumentCoders.h"
#include "RemoteImageBuffer.h"
#include <WebCore/ImageBuffer.h>

namespace WebKit {
using namespace WebCore;

void RemoteResourceCache::cacheNativeImage(Ref<NativeImage>&& image)
{
    m_resourceHeap.add(WTFMove(image));
}

void RemoteResourceCache::cacheDecomposedGlyphs(Ref<DecomposedGlyphs>&& decomposedGlyphs)
{
    m_resourceHeap.add(WTFMove(decomposedGlyphs));
}

void RemoteResourceCache::cacheGradient(Ref<Gradient>&& gradient)
{
    m_resourceHeap.add(WTFMove(gradient));
}

void RemoteResourceCache::cacheFilter(Ref<Filter>&& filter)
{
    m_resourceHeap.add(WTFMove(filter));
}

RefPtr<NativeImage> RemoteResourceCache::cachedNativeImage(RenderingResourceIdentifier renderingResourceIdentifier) const
{
    return m_resourceHeap.getNativeImage(renderingResourceIdentifier);
}

void RemoteResourceCache::cacheFont(Ref<Font>&& font)
{
    m_resourceHeap.add(WTFMove(font));
}

RefPtr<Font> RemoteResourceCache::cachedFont(RenderingResourceIdentifier renderingResourceIdentifier) const
{
    return m_resourceHeap.getFont(renderingResourceIdentifier);
}

void RemoteResourceCache::cacheFontCustomPlatformData(Ref<FontCustomPlatformData>&& customPlatformData)
{
    m_resourceHeap.add(WTFMove(customPlatformData));
}

RefPtr<FontCustomPlatformData> RemoteResourceCache::cachedFontCustomPlatformData(RenderingResourceIdentifier renderingResourceIdentifier) const
{
    return m_resourceHeap.getFontCustomPlatformData(renderingResourceIdentifier);
}

RefPtr<DecomposedGlyphs> RemoteResourceCache::cachedDecomposedGlyphs(RenderingResourceIdentifier renderingResourceIdentifier) const
{
    return m_resourceHeap.getDecomposedGlyphs(renderingResourceIdentifier);
}

RefPtr<Gradient> RemoteResourceCache::cachedGradient(RenderingResourceIdentifier renderingResourceIdentifier) const
{
    return m_resourceHeap.getGradient(renderingResourceIdentifier);
}

RefPtr<Filter> RemoteResourceCache::cachedFilter(RenderingResourceIdentifier renderingResourceIdentifier) const
{
    return m_resourceHeap.getFilter(renderingResourceIdentifier);
}

void RemoteResourceCache::releaseAllResources()
{
    m_resourceHeap.clearAllResources();
}

void RemoteResourceCache::releaseAllDrawingResources()
{
    m_resourceHeap.clearAllDrawingResources();
}

void RemoteResourceCache::releaseAllImageResources()
{
    m_resourceHeap.clearAllImageResources();
}

bool RemoteResourceCache::releaseRenderingResource(RenderingResourceIdentifier renderingResourceIdentifier)
{
    if (m_resourceHeap.removeImageBuffer(renderingResourceIdentifier)
        || m_resourceHeap.removeRenderingResource(renderingResourceIdentifier)
        || m_resourceHeap.removeFont(renderingResourceIdentifier)
        || m_resourceHeap.removeFontCustomPlatformData(renderingResourceIdentifier))
        return true;

    // Caching the remote resource should have happened before releasing it.
    return false;
}

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
