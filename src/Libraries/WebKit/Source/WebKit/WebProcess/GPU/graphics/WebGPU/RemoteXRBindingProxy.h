/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 1, 2025.
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

#include "RemoteDeviceProxy.h"
#include "RemoteGPUProxy.h"
#include "RemotePresentationContextProxy.h"
#include "WebGPUIdentifier.h"
#include <WebCore/WebGPUXRBinding.h>
#include <WebCore/WebGPUXREye.h>

namespace WebCore {
class WebXRFrame;
}

namespace WebCore::WebGPU {
class Device;
class XRProjectionLayer;
class XRView;
}

namespace WebKit::WebGPU {

class ConvertToBackingContext;

class RemoteXRBindingProxy final : public WebCore::WebGPU::XRBinding {
    WTF_MAKE_TZONE_ALLOCATED(RemoteXRBindingProxy);
public:
    static Ref<RemoteXRBindingProxy> create(RemoteDeviceProxy& parent, ConvertToBackingContext& convertToBackingContext, WebGPUIdentifier identifier)
    {
        return adoptRef(*new RemoteXRBindingProxy(parent, convertToBackingContext, identifier));
    }

    virtual ~RemoteXRBindingProxy();

    RemoteDeviceProxy& parent() { return m_parent; }
    RemoteGPUProxy& root() { return m_parent->root(); }
    Ref<RemoteGPUProxy> protectedRoot() { return m_parent->root(); }

private:
    friend class DowncastConvertToBackingContext;

    RemoteXRBindingProxy(RemoteDeviceProxy&, ConvertToBackingContext&, WebGPUIdentifier);

    RemoteXRBindingProxy(const RemoteXRBindingProxy&) = delete;
    RemoteXRBindingProxy(RemoteXRBindingProxy&&) = delete;
    RemoteXRBindingProxy& operator=(const RemoteXRBindingProxy&) = delete;
    RemoteXRBindingProxy& operator=(RemoteXRBindingProxy&&) = delete;

    WebGPUIdentifier backing() const { return m_backing; }

    RefPtr<WebCore::WebGPU::XRProjectionLayer> createProjectionLayer(const WebCore::WebGPU::XRProjectionLayerInit&) final;
    RefPtr<WebCore::WebGPU::XRSubImage> getSubImage(WebCore::WebGPU::XRProjectionLayer&, WebCore::WebXRFrame&, std::optional<WebCore::WebGPU::XREye>/* = "none"*/) final;
    RefPtr<WebCore::WebGPU::XRSubImage> getViewSubImage(WebCore::WebGPU::XRProjectionLayer&) final;
    WebCore::WebGPU::TextureFormat getPreferredColorFormat() final;

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
    Ref<RemoteDeviceProxy> m_parent;
};

} // namespace WebKit::WebGPU

#endif // ENABLE(GPU_PROCESS)
