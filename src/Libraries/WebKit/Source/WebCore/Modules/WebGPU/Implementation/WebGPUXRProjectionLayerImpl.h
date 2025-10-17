/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 17, 2025.
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

#if HAVE(WEBGPU_IMPLEMENTATION)

#include "WebGPUPtr.h"
#include "WebGPUXRProjectionLayer.h"
#include <WebGPU/WebGPU.h>

namespace WebCore {
class Device;
class NativeImage;
}

namespace WebCore::WebGPU {

class ConvertToBackingContext;

class XRProjectionLayerImpl final : public XRProjectionLayer {
    WTF_MAKE_FAST_ALLOCATED;
public:
    static Ref<XRProjectionLayerImpl> create(WebGPUPtr<WGPUXRProjectionLayer>&& projectionLayer, ConvertToBackingContext& convertToBackingContext)
    {
        return adoptRef(*new XRProjectionLayerImpl(WTFMove(projectionLayer), convertToBackingContext));
    }

    virtual ~XRProjectionLayerImpl();
    WGPUXRProjectionLayer backing() const { return m_backing.get(); }

private:
    friend class DowncastConvertToBackingContext;

    explicit XRProjectionLayerImpl(WebGPUPtr<WGPUXRProjectionLayer>&&, ConvertToBackingContext&);

    XRProjectionLayerImpl(const XRProjectionLayerImpl&) = delete;
    XRProjectionLayerImpl(XRProjectionLayerImpl&&) = delete;
    XRProjectionLayerImpl& operator=(const XRProjectionLayerImpl&) = delete;
    XRProjectionLayerImpl& operator=(XRProjectionLayerImpl&&) = delete;

    uint32_t textureWidth() const final;
    uint32_t textureHeight() const final;
    uint32_t textureArrayLength() const final;

    bool ignoreDepthValues() const final;
    std::optional<float> fixedFoveation() const final;
    void setFixedFoveation(std::optional<float>) final;
    WebXRRigidTransform* deltaPose() const final;
    void setDeltaPose(WebXRRigidTransform*) final;

    // WebXRLayer
#if PLATFORM(COCOA)
    void startFrame(size_t frameIndex, MachSendRight&& colorBuffer, MachSendRight&& depthBuffer, MachSendRight&& completionSyncEvent, size_t reusableTextureIndex) final;
#endif
    void endFrame() final;

    WebGPUPtr<WGPUXRProjectionLayer> m_backing;
    Ref<ConvertToBackingContext> m_convertToBackingContext;
#if ENABLE(WEBXR)
    RefPtr<WebXRRigidTransform> m_webXRRigidTransform;
#endif
};

} // namespace WebCore::WebGPU

#endif // HAVE(WEBGPU_IMPLEMENTATION)
