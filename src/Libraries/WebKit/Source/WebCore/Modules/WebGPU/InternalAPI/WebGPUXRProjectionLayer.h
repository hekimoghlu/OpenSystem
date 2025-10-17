/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 15, 2022.
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

#include "PlatformXR.h"
#include "WebGPUTextureFormat.h"
#include "WebGPUTextureUsage.h"
#include "WebGPUXREye.h"
#include "WebGPUXRProjectionLayer.h"
#include "WebGPUXRSubImage.h"

#include <wtf/Ref.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/WeakPtr.h>

namespace WTF {
class MachSendRight;
}

namespace WebCore {
class WebXRRigidTransform;
}

namespace WebCore::WebGPU {

class Device;
class XRGPUSubImage;
class XRProjectionLayer;
class XRFrame;
class XRView;

struct XRProjectionLayerInit {
    TextureFormat colorFormat;
    std::optional<TextureFormat> depthStencilFormat;
    TextureUsageFlags textureUsage { TextureUsage::RenderAttachment };
    double scaleFactor { 1.0 };
};

class XRProjectionLayer : public RefCountedAndCanMakeWeakPtr<XRProjectionLayer> {
public:
    virtual ~XRProjectionLayer() = default;

    virtual uint32_t textureWidth() const = 0;
    virtual uint32_t textureHeight() const = 0;
    virtual uint32_t textureArrayLength() const = 0;

    virtual bool ignoreDepthValues() const = 0;
    virtual std::optional<float> fixedFoveation() const = 0;
    virtual void setFixedFoveation(std::optional<float>) = 0;
    virtual WebXRRigidTransform* deltaPose() const = 0;
    virtual void setDeltaPose(WebXRRigidTransform*) = 0;

    // WebXRLayer
#if PLATFORM(COCOA)
    virtual void startFrame(size_t frameIndex, MachSendRight&& colorBuffer, MachSendRight&& depthBuffer, MachSendRight&& completionSyncEvent, size_t reusableTextureIndex) = 0;
#endif
    virtual void endFrame() = 0;

protected:
    XRProjectionLayer() = default;

private:
    XRProjectionLayer(const XRProjectionLayer&) = delete;
    XRProjectionLayer(XRProjectionLayer&&) = delete;
    XRProjectionLayer& operator=(const XRProjectionLayer&) = delete;
    XRProjectionLayer& operator=(XRProjectionLayer&&) = delete;
};

} // namespace WebCore::WebGPU
