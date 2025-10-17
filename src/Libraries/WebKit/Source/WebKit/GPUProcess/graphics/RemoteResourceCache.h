/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 17, 2023.
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
#pragma once

#if ENABLE(GPU_PROCESS)

#include <WebCore/DisplayListResourceHeap.h>
#include <WebCore/RenderingResourceIdentifier.h>

namespace WebCore {
class ImageBuffer;
}
namespace WebKit {

class RemoteRenderingBackend;

class RemoteResourceCache {
public:
    RemoteResourceCache() = default;

    void cacheNativeImage(Ref<WebCore::NativeImage>&&);
    void cacheFont(Ref<WebCore::Font>&&);
    void cacheDecomposedGlyphs(Ref<WebCore::DecomposedGlyphs>&&);
    void cacheGradient(Ref<WebCore::Gradient>&&);
    void cacheFilter(Ref<WebCore::Filter>&&);
    void cacheFontCustomPlatformData(Ref<WebCore::FontCustomPlatformData>&&);

    const WebCore::DisplayList::ResourceHeap& resourceHeap() const { return m_resourceHeap; }

    RefPtr<WebCore::NativeImage> cachedNativeImage(WebCore::RenderingResourceIdentifier) const;
    RefPtr<WebCore::Font> cachedFont(WebCore::RenderingResourceIdentifier) const;
    RefPtr<WebCore::DecomposedGlyphs> cachedDecomposedGlyphs(WebCore::RenderingResourceIdentifier) const;
    RefPtr<WebCore::Gradient> cachedGradient(WebCore::RenderingResourceIdentifier) const;
    RefPtr<WebCore::Filter> cachedFilter(WebCore::RenderingResourceIdentifier) const;
    RefPtr<WebCore::FontCustomPlatformData> cachedFontCustomPlatformData(WebCore::RenderingResourceIdentifier) const;

    void releaseAllResources();
    void releaseAllDrawingResources();
    void releaseAllImageResources();
    bool releaseRenderingResource(WebCore::RenderingResourceIdentifier);

private:
    WebCore::DisplayList::ResourceHeap m_resourceHeap;
};

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
