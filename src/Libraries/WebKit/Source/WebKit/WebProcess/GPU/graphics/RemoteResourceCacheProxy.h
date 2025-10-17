/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 3, 2024.
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

#include "RenderingUpdateID.h"
#include <WebCore/RenderingResource.h>
#include <wtf/CheckedRef.h>
#include <wtf/HashMap.h>

namespace WebCore {
class DecomposedGlyphs;
class DestinationColorSpace;
class Filter;
class Font;
class Gradient;
class ImageBuffer;
class NativeImage;
struct FontCustomPlatformData;
}

namespace WebKit {

class RemoteImageBufferProxy;
class RemoteRenderingBackendProxy;

class RemoteResourceCacheProxy : public WebCore::RenderingResourceObserver {
public:
    RemoteResourceCacheProxy(RemoteRenderingBackendProxy&);
    ~RemoteResourceCacheProxy();

    void cacheImageBuffer(RemoteImageBufferProxy&);
    RefPtr<RemoteImageBufferProxy> cachedImageBuffer(WebCore::RenderingResourceIdentifier) const;
    void forgetImageBuffer(WebCore::RenderingResourceIdentifier);

    WebCore::NativeImage* cachedNativeImage(WebCore::RenderingResourceIdentifier) const;

    void recordNativeImageUse(WebCore::NativeImage&, const WebCore::DestinationColorSpace&);
    void recordFontUse(WebCore::Font&);
    void recordImageBufferUse(WebCore::ImageBuffer&);
    void recordDecomposedGlyphsUse(WebCore::DecomposedGlyphs&);
    void recordGradientUse(WebCore::Gradient&);
    void recordFilterUse(WebCore::Filter&);
    void recordFontCustomPlatformDataUse(const WebCore::FontCustomPlatformData&);

    void didPaintLayers();

    void remoteResourceCacheWasDestroyed();
    void releaseMemory();
    void releaseAllImageResources();
    
    unsigned imagesCount() const;

    void clear();

private:
    using ImageBufferHashMap = HashMap<WebCore::RenderingResourceIdentifier, ThreadSafeWeakPtr<RemoteImageBufferProxy>>;
    using RenderingResourceHashMap = HashMap<WebCore::RenderingResourceIdentifier, ThreadSafeWeakPtr<WebCore::RenderingResource>>;
    using FontHashMap = HashMap<WebCore::RenderingResourceIdentifier, uint64_t>;

    void releaseRenderingResource(WebCore::RenderingResourceIdentifier) override;
    void clearRenderingResourceMap();
    void clearNativeImageMap();

    void finalizeRenderingUpdateForFonts();
    void prepareForNextRenderingUpdate();
    void clearFontMap();
    void clearFontCustomPlatformDataMap();
    void clearImageBufferBackends();

    ImageBufferHashMap m_imageBuffers;
    RenderingResourceHashMap m_renderingResources;
    FontHashMap m_fonts;
    FontHashMap m_fontCustomPlatformDatas;

    unsigned m_numberOfFontsUsedInCurrentRenderingUpdate { 0 };
    unsigned m_numberOfFontCustomPlatformDatasUsedInCurrentRenderingUpdate { 0 };

    CheckedRef<RemoteRenderingBackendProxy> m_remoteRenderingBackendProxy;
    uint64_t m_renderingUpdateID;
};

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
