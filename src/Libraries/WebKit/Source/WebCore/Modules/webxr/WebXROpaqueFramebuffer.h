/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 14, 2023.
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

#if ENABLE(WEBXR)

#include "GraphicsContextGL.h"
#include "GraphicsTypesGL.h"
#include "PlatformXR.h"
#include "WebXRLayer.h"
#include <wtf/Ref.h>
#include <wtf/RetainPtr.h>
#include <wtf/Vector.h>

namespace WebCore {

class IntSize;
class WebGLFramebuffer;
class WebGLRenderingContextBase;
struct XRWebGLLayerInit;

struct WebXRExternalRenderbuffer {
    GCGLOwnedRenderbuffer renderBufferObject;
    GCGLOwnedExternalImage image;

    explicit operator bool() const { return !!image; }

    void destroyImage(GraphicsContextGL&);
    void release(GraphicsContextGL&);
    void leakObject();
};

template<typename T>
struct WebXRAttachmentSet {
    T colorBuffer;
    T depthStencilBuffer;

    operator bool() const
    {
        return !!colorBuffer; // Need colorBuffer at the minimum!
    }

    void release(GraphicsContextGL& gl)
    {
        colorBuffer.release(gl);
        depthStencilBuffer.release(gl);
    }

    void leakObject()
    {
        colorBuffer.leakObject();
        depthStencilBuffer.leakObject();
    }
};

using WebXRAttachments = WebXRAttachmentSet<GCGLOwnedRenderbuffer>;
using WebXRExternalAttachments = WebXRAttachmentSet<WebXRExternalRenderbuffer>;

class WebXROpaqueFramebuffer {
public:
    struct Attributes {
        bool alpha { true };
        bool antialias { true };
        bool depth { true };
        bool stencil { false };
    };

    static std::unique_ptr<WebXROpaqueFramebuffer> create(PlatformXR::LayerHandle, WebGLRenderingContextBase&, Attributes&&, IntSize);
    ~WebXROpaqueFramebuffer();

    bool supportsDynamicViewportScaling() const;

    PlatformXR::LayerHandle handle() const { return m_handle; }
    const WebGLFramebuffer& framebuffer() const { return m_drawFramebuffer.get(); }
    // Return the size of the framebuffer is Screen Space
    IntSize drawFramebufferSize() const;
    // Return the viewport for eye in Screen Space
    IntRect drawViewport(PlatformXR::Eye) const;

    void startFrame(PlatformXR::FrameData::LayerData&);
    void endFrame();
    bool usesLayeredMode() const;

#if PLATFORM(COCOA)
    void releaseAllDisplayAttachments();
#endif

private:
    WebXROpaqueFramebuffer(PlatformXR::LayerHandle, Ref<WebGLFramebuffer>&&, WebGLRenderingContextBase&, Attributes&&, IntSize);

#if PLATFORM(COCOA)
    bool setupFramebuffer(GraphicsContextGL&, const PlatformXR::FrameData::LayerSetupData&);
    const std::array<WebXRExternalAttachments, 2>* reusableDisplayAttachments(const PlatformXR::FrameData::ExternalTextureData&) const;
    void bindCompositorTexturesForDisplay(GraphicsContextGL&, const PlatformXR::FrameData::LayerData&);
    const std::array<WebXRExternalAttachments, 2>* reusableDisplayAttachmentsAtIndex(size_t);
    void releaseDisplayAttachmentsAtIndex(size_t);
#endif
    void allocateRenderbufferStorage(GraphicsContextGL&, GCGLOwnedRenderbuffer&, GCGLsizei, GCGLenum, IntSize);
    void allocateAttachments(GraphicsContextGL&, WebXRAttachments&, GCGLsizei, IntSize);
    void bindAttachments(GraphicsContextGL&, WebXRAttachments&);
#if PLATFORM(COCOA)
    void bindResolveAttachments(GraphicsContextGL&, WebXRAttachments&);
#endif
    void resolveMSAAFramebuffer(GraphicsContextGL&);
    void blitShared(GraphicsContextGL&);
    void blitSharedToLayered(GraphicsContextGL&);
    IntRect calculateViewportShared(PlatformXR::Eye, bool, const IntRect&, const IntRect&);

    PlatformXR::LayerHandle m_handle;
    Ref<WebGLFramebuffer> m_drawFramebuffer;
    WeakRef<WebGLRenderingContextBase> m_context;
    Attributes m_attributes;
    PlatformXR::Layout m_displayLayout = PlatformXR::Layout::Shared;
    IntSize m_framebufferSize; // Physical Space
#if PLATFORM(COCOA)
    IntRect m_leftViewport; // Screen Space
    IntRect m_rightViewport; // Screen Space
    IntSize m_leftPhysicalSize; // Physical Space
    IntSize m_rightPhysicalSize; // Physical Space
#endif
    WebXRAttachments m_drawAttachments;
    WebXRAttachments m_resolveAttachments;
    GCGLOwnedFramebuffer m_displayFBO;
    GCGLOwnedFramebuffer m_resolvedFBO;
#if PLATFORM(COCOA)
    Vector<std::array<WebXRExternalAttachments, 2>> m_displayAttachmentsSets;
    size_t m_currentDisplayAttachmentIndex { 0 };
    MachSendRight m_completionSyncEvent;
    uint64_t m_renderingFrameIndex { ~0u };
    bool m_usingFoveation { false };
    bool m_blitDepth { false };
#else
    PlatformGLObject m_colorTexture;
#endif
};

} // namespace WebCore

#endif // ENABLE(WEBXR)
