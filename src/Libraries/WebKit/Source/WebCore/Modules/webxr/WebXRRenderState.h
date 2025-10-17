/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 23, 2025.
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

#include "HTMLCanvasElement.h"
#include "WebXRLayer.h"
#include "WebXRWebGLLayer.h"
#include "XRSessionMode.h"

namespace WebCore {

struct XRRenderStateInit;

class WebXRRenderState : public RefCounted<WebXRRenderState> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(WebXRRenderState);
public:
    static Ref<WebXRRenderState> create(XRSessionMode);
    ~WebXRRenderState();

    Ref<WebXRRenderState> clone() const;

    double depthNear() const { return m_depth.near; }
    void setDepthNear(double near) { m_depth.near = near; }

    double depthFar() const { return m_depth.far; }
    void setDepthFar(double far) { m_depth.far = far; };

    std::optional<double> inlineVerticalFieldOfView() const { return m_inlineVerticalFieldOfView; }
    void setInlineVerticalFieldOfView(double fieldOfView) { m_inlineVerticalFieldOfView = fieldOfView; }

    RefPtr<WebXRWebGLLayer> baseLayer() const { return m_baseLayer; }
    void setBaseLayer(WebXRWebGLLayer* baseLayer) { m_baseLayer = baseLayer; }

#if ENABLE(WEBXR_LAYERS)
    const Vector<Ref<WebXRLayer>>& layers() const { return m_layers; }
    void setLayers(const Vector<Ref<WebXRLayer>>&);
#endif

    HTMLCanvasElement* outputCanvas() const { return m_outputCanvas.get(); }
    void setOutputCanvas(HTMLCanvasElement* canvas) { m_outputCanvas = canvas; }

    bool isCompositionEnabled() const { return m_compositionEnabled; }
    void setCompositionEnabled(bool compositionEnabled) { m_compositionEnabled = compositionEnabled; }

private:
    explicit WebXRRenderState(std::optional<double> fieldOfView);
    explicit WebXRRenderState(const WebXRRenderState&);

    // https://immersive-web.github.io/webxr/#initialize-the-render-state
    struct {
        double near { 0.1 }; // in meters
        double far { 1000 }; // in meters
    } m_depth;
    std::optional<double> m_inlineVerticalFieldOfView; // in radians
    RefPtr<WebXRWebGLLayer> m_baseLayer;
#if ENABLE(WEBXR_LAYERS)
    Vector<Ref<WebXRLayer>> m_layers;
#endif
    WeakPtr<HTMLCanvasElement, WeakPtrImplWithEventTargetData> m_outputCanvas;
    bool m_compositionEnabled { true };
};

} // namespace WebCore

#endif // ENABLE(WEBXR)
