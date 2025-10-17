/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 21, 2023.
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

#include "ImageBufferBackendHandleSharing.h"
#include <WebCore/ImageBuffer.h>
#include <wtf/TZoneMalloc.h>

#if USE(CG)
#include <WebCore/ImageBufferCGBackend.h>
#elif USE(CAIRO)
#include <WebCore/ImageBufferCairoBackend.h>
#elif USE(SKIA)
#include <WebCore/ImageBufferSkiaBackend.h>
#endif

namespace WebCore {
class ProcessIdentity;
class ShareableBitmap;
}

namespace WebKit {

#if USE(CG)
using ImageBufferShareableBitmapBackendBase = WebCore::ImageBufferCGBackend;
#elif USE(CAIRO)
using ImageBufferShareableBitmapBackendBase = WebCore::ImageBufferCairoBackend;
#elif USE(SKIA)
using ImageBufferShareableBitmapBackendBase = WebCore::ImageBufferSkiaBackend;
#endif

class ImageBufferShareableBitmapBackend final : public ImageBufferShareableBitmapBackendBase, public ImageBufferBackendHandleSharing {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(ImageBufferShareableBitmapBackend);
    WTF_MAKE_NONCOPYABLE(ImageBufferShareableBitmapBackend);

public:
    static WebCore::IntSize calculateSafeBackendSize(const Parameters&);
    static unsigned calculateBytesPerRow(const Parameters&, const WebCore::IntSize& backendSize);
    static size_t calculateMemoryCost(const Parameters&);

    static std::unique_ptr<ImageBufferShareableBitmapBackend> create(const Parameters&, const WebCore::ImageBufferCreationContext&);
    static std::unique_ptr<ImageBufferShareableBitmapBackend> create(const Parameters&, WebCore::ShareableBitmap::Handle);

    ImageBufferShareableBitmapBackend(const Parameters&, Ref<WebCore::ShareableBitmap>&&, std::unique_ptr<WebCore::GraphicsContext>&&);
    virtual ~ImageBufferShareableBitmapBackend();

    bool canMapBackingStore() const final;
    WebCore::GraphicsContext& context() final { return *m_context; }

    std::optional<ImageBufferBackendHandle> createBackendHandle(WebCore::SharedMemory::Protection = WebCore::SharedMemory::Protection::ReadWrite) const final;
    RefPtr<WebCore::ShareableBitmap> bitmap() const final { return m_bitmap.ptr(); }
#if USE(CAIRO)
    RefPtr<cairo_surface_t> createCairoSurface() final;
#endif
    void transferToNewContext(const WebCore::ImageBufferCreationContext&) final;

    RefPtr<WebCore::NativeImage> copyNativeImage() final;
    RefPtr<WebCore::NativeImage> createNativeImageReference() final;

    void getPixelBuffer(const WebCore::IntRect&, WebCore::PixelBuffer&) final;
    void putPixelBuffer(const WebCore::PixelBuffer&, const WebCore::IntRect& srcRect, const WebCore::IntPoint& destPoint, WebCore::AlphaPremultiplication destFormat) final;

private:
    unsigned bytesPerRow() const final;
    String debugDescription() const final;

    ImageBufferBackendSharing* toBackendSharing() final { return this; }
    void releaseGraphicsContext() final { /* Do nothing. This is only relevant for IOSurface backends */ }

    Ref<WebCore::ShareableBitmap> m_bitmap;
    std::unique_ptr<WebCore::GraphicsContext> m_context;
};

} // namespace WebKit
