/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 2, 2024.
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

#if ENABLE(GPU_PROCESS)

#include "RemoteGPUProxy.h"
#include "WebGPUIdentifier.h"
#include <WebCore/WebGPUXRProjectionLayer.h>

namespace PlatformXR {
struct FrameData;
}

namespace WebCore {
class ImageBuffer;
class NativeImage;
namespace WebGPU {
class Device;
}
}

namespace WebKit::WebGPU {

class ConvertToBackingContext;

class RemoteXRProjectionLayerProxy final : public WebCore::WebGPU::XRProjectionLayer {
    WTF_MAKE_TZONE_ALLOCATED(RemoteXRProjectionLayerProxy);
public:
    static Ref<RemoteXRProjectionLayerProxy> create(Ref<RemoteGPUProxy>&& parent, ConvertToBackingContext& convertToBackingContext, WebGPUIdentifier identifier)
    {
        return adoptRef(*new RemoteXRProjectionLayerProxy(WTFMove(parent), convertToBackingContext, identifier));
    }

    virtual ~RemoteXRProjectionLayerProxy();

    RemoteGPUProxy& parent() { return m_parent; }
    RemoteGPUProxy& root() { return m_parent; }
    WebGPUIdentifier backing() const { return m_backing; }

private:
    friend class DowncastConvertToBackingContext;

    RemoteXRProjectionLayerProxy(Ref<RemoteGPUProxy>&&, ConvertToBackingContext&, WebGPUIdentifier);

    RemoteXRProjectionLayerProxy(const RemoteXRProjectionLayerProxy&) = delete;
    RemoteXRProjectionLayerProxy(RemoteXRProjectionLayerProxy&&) = delete;
    RemoteXRProjectionLayerProxy& operator=(const RemoteXRProjectionLayerProxy&) = delete;
    RemoteXRProjectionLayerProxy& operator=(RemoteXRProjectionLayerProxy&&) = delete;

    uint32_t textureWidth() const final;
    uint32_t textureHeight() const final;
    uint32_t textureArrayLength() const final;

    bool ignoreDepthValues() const final;
    std::optional<float> fixedFoveation() const final;
    void setFixedFoveation(std::optional<float>) final;
    WebCore::WebXRRigidTransform* deltaPose() const final;
    void setDeltaPose(WebCore::WebXRRigidTransform*) final;

    // WebXRLayer
#if PLATFORM(COCOA)
    void startFrame(size_t frameIndex, MachSendRight&&, MachSendRight&&, MachSendRight&&, size_t reusableTextureIndex) final;
#endif
    void endFrame() final;

    template<typename T>
    WARN_UNUSED_RETURN IPC::Error send(T&& message)
    {
        return root().protectedStreamClientConnection()->send(WTFMove(message), backing());
    }
    template<typename T>
    WARN_UNUSED_RETURN IPC::Connection::SendSyncResult<T> sendSync(T&& message)
    {
        return root().protectedStreamClientConnection()->sendSync(WTFMove(message), backing());
    }

    WebGPUIdentifier m_backing;
    Ref<ConvertToBackingContext> m_convertToBackingContext;
    Ref<RemoteGPUProxy> m_parent;
};

} // namespace WebKit::WebGPU

#endif // ENABLE(GPU_PROCESS)
