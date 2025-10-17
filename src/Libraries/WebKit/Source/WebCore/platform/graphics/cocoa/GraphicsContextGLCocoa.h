/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 26, 2024.
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

#if ENABLE(WEBGL)

#include "GraphicsContextGLANGLE.h"
#include "IOSurfaceDrawingBuffer.h"
#include "ProcessIdentity.h"
#include <array>

#if ENABLE(MEDIA_STREAM)
#include <memory>
#endif

#if ENABLE(WEBXR)
#include "PlatformXR.h"
#include <wtf/EnumeratedArray.h>
#endif

OBJC_CLASS MTLSharedEventListener;
#if ENABLE(WEBXR)
OBJC_PROTOCOL(MTLRasterizationRateMap);
#endif

namespace WebCore {

class GraphicsLayerContentsDisplayDelegate;

#if ENABLE(VIDEO)
class GraphicsContextGLCVCocoa;
#endif

#if ENABLE(MEDIA_STREAM)
class ImageRotationSessionVT;
#endif

// IOSurface backing store for an image of a texture.
// When preserveDrawingBuffer == false, this is the drawing buffer backing store.
// When preserveDrawingBuffer == true, this is blitted to during display prepare.
class IOSurfacePbuffer : public IOSurfaceDrawingBuffer {
public:
    IOSurfacePbuffer() = default;
    IOSurfacePbuffer(IOSurfacePbuffer&&);
    explicit IOSurfacePbuffer(std::unique_ptr<IOSurface>&&, void* pbuffer);
    void* pbuffer() const { return m_pbuffer; }
    IOSurfacePbuffer& operator=(IOSurfacePbuffer&&);
private:
    void* m_pbuffer { nullptr };
};

class WEBCORE_EXPORT GraphicsContextGLCocoa : public GraphicsContextGLANGLE {
public:
    static RefPtr<GraphicsContextGLCocoa> create(WebCore::GraphicsContextGLAttributes&&, ProcessIdentity&& resourceOwner);
    ~GraphicsContextGLCocoa();
    IOSurface* displayBufferSurface();

    std::tuple<GCGLenum, GCGLenum> externalImageTextureBindingPoint() final;

    enum class PbufferAttachmentUsage { Read, Write, ReadWrite };
    // Returns a handle which, if non-null, must be released via the
    // detach call below.
    void* createPbufferAndAttachIOSurface(GCGLenum target, PbufferAttachmentUsage, GCGLenum internalFormat, GCGLsizei width, GCGLsizei height, GCGLenum type, IOSurfaceRef, GCGLuint plane);
    void destroyPbufferAndDetachIOSurface(void* handle);

#if ENABLE(WEBXR)
    GCGLExternalImage createExternalImage(ExternalImageSource&&, GCGLenum internalFormat, GCGLint layer) final;
    void bindExternalImage(GCGLenum target, GCGLExternalImage) final;

    bool addFoveation(IntSize, IntSize, IntSize, std::span<const GCGLfloat>, std::span<const GCGLfloat>, std::span<const GCGLfloat>) final;
    void enableFoveation(GCGLuint) final;
    void disableFoveation() final;

    RetainPtr<id> newSharedEventWithMachPort(mach_port_t);
    GCGLExternalSync createExternalSync(ExternalSyncSource&&) final;
#endif
    GCGLExternalSync createExternalSync(id, uint64_t);

#if ENABLE(WEBXR)
    bool enableRequiredWebXRExtensions() final;

    // GL_EXT_discard_framebuffer
    void framebufferDiscard(GCGLenum, std::span<const GCGLenum>) final;
    // GL_WEBKIT_explicit_resolve_target
    void framebufferResolveRenderbuffer(GCGLenum, GCGLenum, GCGLenum, PlatformGLObject) final;
#endif

    void waitUntilWorkScheduled();

    // GraphicsContextGLANGLE overrides.
    RefPtr<GraphicsLayerContentsDisplayDelegate> layerContentsDisplayDelegate() override;
#if ENABLE(VIDEO)
    bool copyTextureFromVideoFrame(VideoFrame&, PlatformGLObject texture, uint32_t target, int32_t level, uint32_t internalFormat, uint32_t format, uint32_t type, bool premultiplyAlpha, bool flipY) final;
#endif
#if ENABLE(MEDIA_STREAM) || ENABLE(WEB_CODECS)
    RefPtr<VideoFrame> surfaceBufferToVideoFrame(SurfaceBuffer) final;
#endif
    RefPtr<PixelBuffer> readCompositedResults() final;
    void setDrawingBufferColorSpace(const DestinationColorSpace&) final;
    void prepareForDisplay() override;

    RefPtr<NativeImage> bufferAsNativeImage(SurfaceBuffer) override;

    // Prepares current frame for display. The `finishedSignal` will be invoked once the frame has finished rendering.
    void prepareForDisplayWithFinishedSignal(Function<void()> finishedSignal);

protected:
    GraphicsContextGLCocoa(WebCore::GraphicsContextGLAttributes&&, ProcessIdentity&& resourceOwner);

    // GraphicsContextGLANGLE overrides.
    bool platformInitializeContext() final;
    bool platformInitializeExtensions() final;
    bool platformInitialize() final;
    void invalidateKnownTextureContent(GCGLuint) final;
    bool reshapeDrawingBuffer() final;
    void prepareForDrawingBufferWrite() final;

    IOSurfacePbuffer& drawingBuffer();
    IOSurfacePbuffer& displayBuffer();
    IOSurfacePbuffer& surfaceBuffer(SurfaceBuffer);
    bool bindNextDrawingBuffer();
    void freeDrawingBuffers();

    // Inserts new fence that will invoke `signal` from a background thread when completed.
    // If not possible, calls the `signal`.
    void insertFinishedSignalOrInvoke(Function<void()> signal);
#if ENABLE(WEBXR)
    bool enableRequiredWebXRExtensionsImpl();
#endif
#if ENABLE(VIDEO)
    GraphicsContextGLCV* cvContext();
#endif

    ProcessIdentity m_resourceOwner;
    DestinationColorSpace m_drawingBufferColorSpace;
#if ENABLE(VIDEO)
    std::unique_ptr<GraphicsContextGLCVCocoa> m_cv;
#endif
#if ENABLE(MEDIA_STREAM)
    std::unique_ptr<ImageRotationSessionVT> m_mediaSampleRotationSession;
    IntSize m_mediaSampleRotationSessionSize;
#endif
    RetainPtr<MTLSharedEventListener> m_finishedMetalSharedEventListener;
    RetainPtr<id> m_finishedMetalSharedEvent; // FIXME: Remove all C++ includees and use id<MTLSharedEvent>.
#if ENABLE(WEBXR)
    using RasterizationRateMapArray =  EnumeratedArray<PlatformXR::Layout, RetainPtr<MTLRasterizationRateMap>, PlatformXR::Layout::Layered>;
    RasterizationRateMapArray m_rasterizationRateMap;
#endif

    static constexpr size_t maxReusedDrawingBuffers { 3 };
    size_t m_currentDrawingBufferIndex { 0 };
    std::array<IOSurfacePbuffer, maxReusedDrawingBuffers> m_drawingBuffers;
    friend class GraphicsContextGLCVCocoa;
};

inline IOSurfacePbuffer::IOSurfacePbuffer(std::unique_ptr<IOSurface>&& surface, void* pbuffer)
    : IOSurfaceDrawingBuffer(WTFMove(surface))
    , m_pbuffer(pbuffer)
{
}

inline IOSurfacePbuffer::IOSurfacePbuffer(IOSurfacePbuffer&& other)
    : IOSurfaceDrawingBuffer(WTFMove(other))
    , m_pbuffer(std::exchange(other.m_pbuffer, nullptr))
{
}

inline IOSurfacePbuffer& IOSurfacePbuffer::operator=(IOSurfacePbuffer&& other)
{
    IOSurfaceDrawingBuffer::operator=(WTFMove(other));
    m_pbuffer = std::exchange(other.m_pbuffer, nullptr);
    return *this;
}

}

#endif
