/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 22, 2023.
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

#if ENABLE(GPU_PROCESS) && HAVE(IOSURFACE)

#include "ImageBufferBackendHandleSharing.h"
#include <WebCore/IOSurface.h>
#include <WebCore/ImageBufferIOSurfaceBackend.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {
class ProcessIdentity;
}

namespace WebKit {

// ImageBufferBackend for small LayerBacking stores.
class ImageBufferShareableMappedIOSurfaceBitmapBackend final : public WebCore::ImageBufferCGBackend, public ImageBufferBackendHandleSharing {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(ImageBufferShareableMappedIOSurfaceBitmapBackend);
    WTF_MAKE_NONCOPYABLE(ImageBufferShareableMappedIOSurfaceBitmapBackend);
public:
    static std::unique_ptr<ImageBufferShareableMappedIOSurfaceBitmapBackend> create(const Parameters&, const WebCore::ImageBufferCreationContext&);
    static size_t calculateMemoryCost(const Parameters& parameters) { return WebCore::ImageBufferIOSurfaceBackend::calculateMemoryCost(parameters); }

    ImageBufferShareableMappedIOSurfaceBitmapBackend(const Parameters&, std::unique_ptr<WebCore::IOSurface>, WebCore::IOSurface::LockAndContext&&, WebCore::IOSurfacePool*);
    ~ImageBufferShareableMappedIOSurfaceBitmapBackend();

    static constexpr WebCore::RenderingMode renderingMode = WebCore::RenderingMode::Accelerated;
    bool canMapBackingStore() const final;

    std::optional<ImageBufferBackendHandle> createBackendHandle(WebCore::SharedMemory::Protection = WebCore::SharedMemory::Protection::ReadWrite) const final;
    WebCore::GraphicsContext& context() final;
private:
    // ImageBufferBackendSharing
    ImageBufferBackendSharing* toBackendSharing() final { return this; }

    // WebCore::ImageBufferCGBackend
    unsigned bytesPerRow() const final;
    RefPtr<WebCore::NativeImage> copyNativeImage() final;
    RefPtr<WebCore::NativeImage> createNativeImageReference() final;
    RefPtr<WebCore::NativeImage> sinkIntoNativeImage() final;
    bool isInUse() const final;
    void releaseGraphicsContext() final;
    bool setVolatile() final;
    WebCore::SetNonVolatileResult setNonVolatile() final;
    WebCore::VolatilityState volatilityState() const final;
    void setVolatilityState(WebCore::VolatilityState) final;
    void transferToNewContext(const WebCore::ImageBufferCreationContext&) final;
    void getPixelBuffer(const WebCore::IntRect&, WebCore::PixelBuffer&) final;
    void putPixelBuffer(const WebCore::PixelBuffer&, const WebCore::IntRect&, const WebCore::IntPoint&, WebCore::AlphaPremultiplication) final;
    void flushContext() final;

    std::unique_ptr<WebCore::IOSurface> m_surface;
    std::optional<WebCore::IOSurface::Locker<WebCore::IOSurface::AccessMode::ReadWrite>> m_lock;
    WebCore::VolatilityState m_volatilityState { WebCore::VolatilityState::NonVolatile };
    RefPtr<WebCore::IOSurfacePool> m_ioSurfacePool;
};

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS) && HAVE(IOSURFACE)
