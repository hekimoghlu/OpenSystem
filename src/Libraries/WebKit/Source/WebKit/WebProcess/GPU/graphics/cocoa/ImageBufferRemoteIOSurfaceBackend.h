/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 20, 2022.
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

#if HAVE(IOSURFACE)

#include "ImageBufferBackendHandleSharing.h"
#include <WebCore/GraphicsContext.h>
#include <WebCore/ImageBufferBackend.h>
#include <wtf/TZoneMalloc.h>

namespace WebKit {

class ImageBufferRemoteIOSurfaceBackend final : public WebCore::ImageBufferBackend, public ImageBufferBackendHandleSharing {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(ImageBufferRemoteIOSurfaceBackend);
    WTF_MAKE_NONCOPYABLE(ImageBufferRemoteIOSurfaceBackend);
public:
    static WebCore::IntSize calculateSafeBackendSize(const Parameters&);
    static size_t calculateMemoryCost(const Parameters&);

    static std::unique_ptr<ImageBufferRemoteIOSurfaceBackend> create(const Parameters&, ImageBufferBackendHandle);

    ImageBufferRemoteIOSurfaceBackend(const Parameters& parameters, MachSendRight&& handle)
        : ImageBufferBackend(parameters)
        , m_handle(WTFMove(handle))
    {
    }

    static constexpr WebCore::RenderingMode renderingMode = WebCore::RenderingMode::Accelerated;
    bool canMapBackingStore() const final;

    WebCore::GraphicsContext& context() final;
    std::optional<ImageBufferBackendHandle> createBackendHandle(WebCore::SharedMemory::Protection = WebCore::SharedMemory::Protection::ReadWrite) const final;
    std::optional<ImageBufferBackendHandle> takeBackendHandle(WebCore::SharedMemory::Protection = WebCore::SharedMemory::Protection::ReadWrite) final;

private:
    RefPtr<WebCore::NativeImage> copyNativeImage() final;
    RefPtr<WebCore::NativeImage> createNativeImageReference() final;

    void getPixelBuffer(const WebCore::IntRect&, WebCore::PixelBuffer&) final;
    void putPixelBuffer(const WebCore::PixelBuffer&, const WebCore::IntRect& srcRect, const WebCore::IntPoint& destPoint, WebCore::AlphaPremultiplication destFormat) final;

    unsigned bytesPerRow() const final;

    WebCore::VolatilityState volatilityState() const final { return m_volatilityState; }
    void setVolatilityState(WebCore::VolatilityState volatilityState) final { m_volatilityState = volatilityState; }

    // ImageBufferBackendSharing
    ImageBufferBackendSharing* toBackendSharing() final { return this; }
    void setBackendHandle(ImageBufferBackendHandle&&) final;
    void clearBackendHandle() final;

    String debugDescription() const final;

    MachSendRight m_handle;

    WebCore::VolatilityState m_volatilityState { WebCore::VolatilityState::NonVolatile };
};

} // namespace WebKit

#endif // HAVE(IOSURFACE)
