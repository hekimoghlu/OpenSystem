/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 5, 2022.
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

#include "ImageBuffer.h"
#include "ImageBufferCGBackend.h"
#include "IOSurface.h"
#include "IOSurfacePool.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class WEBCORE_EXPORT ImageBufferIOSurfaceBackend : public ImageBufferCGBackend {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED_EXPORT(ImageBufferIOSurfaceBackend, WEBCORE_EXPORT);
    WTF_MAKE_NONCOPYABLE(ImageBufferIOSurfaceBackend);
public:
    static IntSize calculateSafeBackendSize(const Parameters&);
    static unsigned calculateBytesPerRow(const IntSize& backendSize);
    static size_t calculateMemoryCost(const Parameters&);

    static std::unique_ptr<ImageBufferIOSurfaceBackend> create(const Parameters&, const ImageBufferCreationContext&);

    ~ImageBufferIOSurfaceBackend();
    
    static constexpr RenderingMode renderingMode = RenderingMode::Accelerated;
    bool canMapBackingStore() const final;

    IOSurface* surface() override;
    GraphicsContext& context() override;
    void flushContext() override;

protected:
    ImageBufferIOSurfaceBackend(const Parameters&, std::unique_ptr<IOSurface>, RetainPtr<CGContextRef> platformContext, PlatformDisplayID, IOSurfacePool*);
    CGContextRef ensurePlatformContext();
    // Returns true if flush happened.
    bool flushContextDraws();
    
    RefPtr<NativeImage> copyNativeImage() override;
    RefPtr<NativeImage> createNativeImageReference() override;
    RefPtr<NativeImage> sinkIntoNativeImage() override;

    void getPixelBuffer(const IntRect&, PixelBuffer&) override;
    void putPixelBuffer(const PixelBuffer&, const IntRect& srcRect, const IntPoint& destPoint, AlphaPremultiplication destFormat) override;

    bool isInUse() const override;
    void releaseGraphicsContext() override;

    bool setVolatile() final;
    SetNonVolatileResult setNonVolatile() final;
    VolatilityState volatilityState() const final;
    void setVolatilityState(VolatilityState) final;

    void ensureNativeImagesHaveCopiedBackingStore() final;

    void transferToNewContext(const ImageBufferCreationContext&) final;

    unsigned bytesPerRow() const override;

    // Returns true if this invalidation requires a flush to complete
    bool invalidateCachedNativeImage();
    void prepareForExternalRead();
    void prepareForExternalWrite();

    RetainPtr<CGImageRef> createImage();
    RetainPtr<CGImageRef> createImageReference();

    std::unique_ptr<IOSurface> m_surface;
    RetainPtr<CGContextRef> m_platformContext;
    const PlatformDisplayID m_displayID;
    bool m_mayHaveOutstandingBackingStoreReferences { false };
    VolatilityState m_volatilityState { VolatilityState::NonVolatile };
    RefPtr<IOSurfacePool> m_ioSurfacePool;
    bool m_needsFirstFlush { true };
};

} // namespace WebCore

#endif // HAVE(IOSURFACE)
