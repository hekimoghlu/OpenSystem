/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 11, 2023.
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

#if ENABLE(WEBXR_LAYERS)

#include "GPUTextureFormat.h"
#include "WebGPUXRBinding.h"
#include "WebXRSession.h"
#include "XREye.h"
#include "XRGPUProjectionLayerInit.h"

#include <ExceptionOr.h>
#include <wtf/Ref.h>
#include <wtf/RefPtr.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

namespace WebGPU {
class XRBinding;
}

enum class GPUTextureFormat : uint8_t;

class GPUDevice;
class WebXRFrame;
class WebXRSession;
class WebXRView;
class XRCompositionLayer;
class XRCubeLayer;
class XRCylinderLayer;
class XREquirectLayer;
class XRProjectionLayer;
class XRQuadLayer;
class XRGPUSubImage;

struct XRCubeLayerInit;
struct XRCylinderLayerInit;
struct XREquirectLayerInit;
struct XRGPUProjectionLayerInit;
struct XRProjectionLayerInit;
struct XRQuadLayerInit;

// https://github.com/immersive-web/WebXR-WebGPU-Binding/blob/main/explainer.md
class XRGPUBinding : public RefCounted<XRGPUBinding> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(XRGPUBinding);
public:
    static Ref<XRGPUBinding> create(const WebXRSession& session, GPUDevice& device)
    {
        return adoptRef(*new XRGPUBinding(session, device));
    }

    double nativeProjectionScaleFactor() const;

    ExceptionOr<Ref<XRProjectionLayer>> createProjectionLayer(ScriptExecutionContext&, std::optional<XRGPUProjectionLayerInit>);
    RefPtr<XRGPUSubImage> getSubImage(XRCompositionLayer&, WebXRFrame&, std::optional<XREye>/* = "none"*/);
    ExceptionOr<Ref<XRGPUSubImage>> getViewSubImage(XRProjectionLayer&, WebXRView&);
    GPUTextureFormat getPreferredColorFormat();

    GPUDevice& device();

    // The core specification doesn't require these, support will be added later.
    // XRQuadLayer createQuadLayer(optional XRGPUQuadLayerInit init);
    // XRCylinderLayer createCylinderLayer(optional XRGPUCylinderLayerInit init);
    // XREquirectLayer createEquirectLayer(optional XRGPUEquirectLayerInit init);
    // XRCubeLayer createCubeLayer(optional XRGPUCubeLayerInit init);
private:
    XRGPUBinding(const WebXRSession&, GPUDevice&);

    RefPtr<WebGPU::XRBinding> m_backing;
    RefPtr<const WebXRSession> m_session;
    std::optional<XRGPUProjectionLayerInit> m_init;
    Ref<GPUDevice> m_device;
};

} // namespace WebCore

#endif // ENABLE(WEBXR_LAYERS)
