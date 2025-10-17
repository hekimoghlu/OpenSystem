/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 18, 2025.
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

#include "CanvasObserver.h"
#include "ExceptionOr.h"
#include "FloatRect.h"
#include "GraphicsTypesGL.h"
#include "PlatformXR.h"
#include "WebXRLayer.h"
#include <variant>
#include <wtf/Ref.h>
#include <wtf/RefPtr.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class HTMLCanvasElement;
class IntSize;
class WebGLFramebuffer;
class WebGLRenderingContext;
class WebGLRenderingContextBase;
class WebGL2RenderingContext;
class WebXROpaqueFramebuffer;
class WebXRSession;
class WebXRView;
class WebXRViewport;
struct XRWebGLLayerInit;

class WebXRWebGLLayer : public WebXRLayer, private CanvasObserver {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(WebXRWebGLLayer);
public:

    using WebXRRenderingContext = std::variant<
        RefPtr<WebGLRenderingContext>,
        RefPtr<WebGL2RenderingContext>
    >;

    static ExceptionOr<Ref<WebXRWebGLLayer>> create(Ref<WebXRSession>&&, WebXRRenderingContext&&, const XRWebGLLayerInit&);
    ~WebXRWebGLLayer();

    bool antialias() const;
    bool ignoreDepthValues() const;

    const WebGLFramebuffer* framebuffer() const;
    unsigned framebufferWidth() const;
    unsigned framebufferHeight() const;

    ExceptionOr<RefPtr<WebXRViewport>> getViewport(WebXRView&);

    static double getNativeFramebufferScaleFactor(const WebXRSession&);

    const WebXRSession* session() { return m_session.get(); }

    bool isCompositionEnabled() const { return m_isCompositionEnabled; }

    HTMLCanvasElement* canvas() const;

    void sessionEnded();

    // WebXRLayer
    void startFrame(PlatformXR::FrameData&) final;
    PlatformXR::Device::Layer endFrame() final;

private:
    WebXRWebGLLayer(Ref<WebXRSession>&&, WebXRRenderingContext&&, std::unique_ptr<WebXROpaqueFramebuffer>&&, bool antialias, bool ignoreDepthValues, bool isCompositionEnabled);

    void computeViewports();
    static IntSize computeNativeWebGLFramebufferResolution();
    static IntSize computeRecommendedWebGLFramebufferResolution();

    void canvasChanged(CanvasBase&, const FloatRect&) final { };
    void canvasResized(CanvasBase&) final;
    void canvasDestroyed(CanvasBase&) final { };
    RefPtr<WebXRSession> m_session;
    WebXRRenderingContext m_context;

    struct ViewportData {
        Ref<WebXRViewport> viewport;
        double currentScale { 1.0 };
    };

    ViewportData m_leftViewportData;
    ViewportData m_rightViewportData;
    std::unique_ptr<WebXROpaqueFramebuffer> m_framebuffer;
    bool m_antialias { false };
    bool m_ignoreDepthValues { false };
    bool m_isCompositionEnabled { true };
};

} // namespace WebCore

#endif // ENABLE(WEBXR)
