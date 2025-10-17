/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 17, 2023.
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

#include "CopyImageOptions.h"
#include "DestinationColorSpace.h"
#include "FloatRect.h"
#include "GraphicsLayerContentsDisplayDelegate.h"
#include "GraphicsTypesGL.h"
#include "ImageBufferAllocator.h"
#include "ImageBufferBackendParameters.h"
#include "ImagePaintingOptions.h"
#include "IntRect.h"
#include "PixelBufferFormat.h"
#include "PlatformLayer.h"
#include "RenderingMode.h"
#include <wtf/RefPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

#if USE(CAIRO)
#include "RefPtrCairo.h"
#include <cairo.h>
#endif

#if HAVE(IOSURFACE)
#include "IOSurface.h"
#endif

#if USE(SKIA)
class GrDirectContext;
#endif

namespace WTF {
class TextStream;
}

namespace WebCore {

struct ImageBufferCreationContext;
class GraphicsContext;
class GraphicsContextGL;
#if HAVE(IOSURFACE)
class IOSurfacePool;
#endif
class Image;
class NativeImage;
class PixelBuffer;
class ProcessIdentity;
class SharedBuffer;

enum class PreserveResolution : bool {
    No,
    Yes,
};

enum class SetNonVolatileResult : uint8_t {
    Valid,
    Empty
};

enum class VolatilityState : uint8_t {
    NonVolatile,
    Volatile
};

class ThreadSafeImageBufferFlusher {
    WTF_MAKE_TZONE_ALLOCATED(ThreadSafeImageBufferFlusher);
    WTF_MAKE_NONCOPYABLE(ThreadSafeImageBufferFlusher);
public:
    ThreadSafeImageBufferFlusher() = default;
    virtual ~ThreadSafeImageBufferFlusher() = default;
    virtual void flush() = 0;
};

class ImageBufferBackendSharing {
public:
    virtual ~ImageBufferBackendSharing() = default;
    virtual bool isImageBufferBackendHandleSharing() const { return false; }
};

class ImageBufferBackend {
public:
    using Parameters = ImageBufferBackendParameters;

    struct Info {
        RenderingMode renderingMode;
        AffineTransform baseTransform;
        size_t memoryCost;
    };

    WEBCORE_EXPORT virtual ~ImageBufferBackend();

    WEBCORE_EXPORT static IntSize calculateSafeBackendSize(const Parameters&);
    WEBCORE_EXPORT static size_t calculateMemoryCost(const IntSize& backendSize, unsigned bytesPerRow);
    WEBCORE_EXPORT static AffineTransform calculateBaseTransform(const Parameters&);

    virtual GraphicsContext& context() = 0;
    virtual void flushContext() { }

    virtual RefPtr<NativeImage> copyNativeImage() = 0;
    virtual RefPtr<NativeImage> createNativeImageReference() = 0;
    WEBCORE_EXPORT virtual RefPtr<NativeImage> sinkIntoNativeImage();

    WEBCORE_EXPORT void convertToLuminanceMask();
    virtual void transformToColorSpace(const DestinationColorSpace&) { }

    virtual void getPixelBuffer(const IntRect& srcRect, PixelBuffer& destination) = 0;
    virtual void putPixelBuffer(const PixelBuffer&, const IntRect& srcRect, const IntPoint& destPoint, AlphaPremultiplication destFormat) = 0;

    WEBCORE_EXPORT virtual RefPtr<SharedBuffer> sinkIntoPDFDocument();

#if HAVE(IOSURFACE)
    virtual IOSurface* surface() { return nullptr; }
#endif

#if USE(CAIRO)
    virtual RefPtr<cairo_surface_t> createCairoSurface() { return nullptr; }
#endif

#if USE(SKIA)
    virtual void finishAcceleratedRenderingAndCreateFence() { }
    virtual void waitForAcceleratedRenderingFenceCompletion() { }

    virtual const GrDirectContext* skiaGrContext() const { return nullptr; }
    WEBCORE_EXPORT virtual RefPtr<ImageBuffer> copyAcceleratedImageBufferBorrowingBackendRenderTarget(const ImageBuffer&) const;
#endif

    virtual bool isInUse() const { return false; }
    virtual void releaseGraphicsContext() { ASSERT_NOT_REACHED(); }

    virtual void transferToNewContext(const ImageBufferCreationContext&) { }

    // Returns true on success.
    virtual bool setVolatile() { return true; }
    virtual SetNonVolatileResult setNonVolatile() { return SetNonVolatileResult::Valid; }
    virtual VolatilityState volatilityState() const { return VolatilityState::NonVolatile; }
    virtual void setVolatilityState(VolatilityState) { }

    virtual std::unique_ptr<ThreadSafeImageBufferFlusher> createFlusher() { return nullptr; }

    static constexpr RenderingMode renderingMode = RenderingMode::Unaccelerated;

    virtual bool canMapBackingStore() const = 0;
    virtual void ensureNativeImagesHaveCopiedBackingStore() { }

    virtual ImageBufferBackendSharing* toBackendSharing() { return nullptr; }

    virtual RefPtr<GraphicsLayerContentsDisplayDelegate> layerContentsDisplayDelegate() const { return nullptr; }

    virtual void prepareForDisplay() { }

    const Parameters& parameters() const { return m_parameters; }

    WEBCORE_EXPORT virtual String debugDescription() const = 0;

protected:
    WEBCORE_EXPORT ImageBufferBackend(const Parameters&);

    virtual unsigned bytesPerRow() const = 0;

    IntSize size() const { return m_parameters.backendSize; };
    float resolutionScale() const { return m_parameters.resolutionScale; }
    const DestinationColorSpace& colorSpace() const { return m_parameters.colorSpace; }
    ImageBufferPixelFormat pixelFormat() const { return m_parameters.pixelFormat; }

    WEBCORE_EXPORT void getPixelBuffer(const IntRect& srcRect, std::span<const uint8_t> data, PixelBuffer& destination);
    WEBCORE_EXPORT void putPixelBuffer(const PixelBuffer&, const IntRect& srcRect, const IntPoint& destPoint, AlphaPremultiplication destFormat, std::span<uint8_t> destination);

    Parameters m_parameters;
};

WEBCORE_EXPORT TextStream& operator<<(TextStream&, VolatilityState);
WEBCORE_EXPORT TextStream& operator<<(TextStream&, const ImageBufferBackend&);

} // namespace WebCore
