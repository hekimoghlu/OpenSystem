/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 9, 2022.
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

#include "WebGPUXRProjectionLayer.h"
#include "WebXRRigidTransform.h"
#include "XRCompositionLayer.h"

#include <wtf/MachSendRight.h>

namespace WebCore {

namespace WebGPU {
class XRProjectionLayer;
}

class GPUTexture;

class XRProjectionLayer : public XRCompositionLayer {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(XRProjectionLayer);
public:
    static Ref<XRProjectionLayer> create(ScriptExecutionContext& scriptExecutionContext, Ref<WebCore::WebGPU::XRProjectionLayer>&& backing)
    {
        return adoptRef(*new XRProjectionLayer(scriptExecutionContext, WTFMove(backing)));
    }
    virtual ~XRProjectionLayer();

    uint32_t textureWidth() const;
    uint32_t textureHeight() const;
    uint32_t textureArrayLength() const;

    bool ignoreDepthValues() const;
    std::optional<float> fixedFoveation() const;
    [[noreturn]] void setFixedFoveation(std::optional<float>);
    WebXRRigidTransform* deltaPose() const;
    [[noreturn]] void setDeltaPose(WebXRRigidTransform*);

    // WebXRLayer
    void startFrame(PlatformXR::FrameData&) final;
    PlatformXR::Device::Layer endFrame() final;

    WebCore::WebGPU::XRProjectionLayer& backing();
private:
    XRProjectionLayer(ScriptExecutionContext&, Ref<WebCore::WebGPU::XRProjectionLayer>&&);

    Ref<WebCore::WebGPU::XRProjectionLayer> m_backing;
};

} // namespace WebCore

#endif
