/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 19, 2023.
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
#include <WebCore/WebGPUXRSubImage.h>

namespace WebCore {
class ImageBuffer;
class NativeImage;
namespace WebGPU {
class Device;
}
}

namespace WebKit::WebGPU {

class ConvertToBackingContext;
class RemoteTextureProxy;

class RemoteXRSubImageProxy final : public WebCore::WebGPU::XRSubImage {
    WTF_MAKE_TZONE_ALLOCATED(RemoteXRSubImageProxy);
public:
    static Ref<RemoteXRSubImageProxy> create(Ref<RemoteGPUProxy>&& parent, ConvertToBackingContext& convertToBackingContext, WebGPUIdentifier identifier)
    {
        return adoptRef(*new RemoteXRSubImageProxy(WTFMove(parent), convertToBackingContext, identifier));
    }

    virtual ~RemoteXRSubImageProxy();

    RemoteGPUProxy& parent() { return m_parent; }
    RemoteGPUProxy& root() { return m_parent; }
    Ref<RemoteGPUProxy> protectedRoot() { return m_parent; }

private:
    friend class DowncastConvertToBackingContext;

    RemoteXRSubImageProxy(Ref<RemoteGPUProxy>&&, ConvertToBackingContext&, WebGPUIdentifier);

    RemoteXRSubImageProxy(const RemoteXRSubImageProxy&) = delete;
    RemoteXRSubImageProxy(RemoteXRSubImageProxy&&) = delete;
    RemoteXRSubImageProxy& operator=(const RemoteXRSubImageProxy&) = delete;
    RemoteXRSubImageProxy& operator=(RemoteXRSubImageProxy&&) = delete;

    WebGPUIdentifier backing() const { return m_backing; }
    RefPtr<WebCore::WebGPU::Texture> colorTexture() final;
    RefPtr<WebCore::WebGPU::Texture> depthStencilTexture() final;
    RefPtr<WebCore::WebGPU::Texture> motionVectorTexture() final;

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

    RefPtr<RemoteTextureProxy> m_currentTexture;
    RefPtr<RemoteTextureProxy> m_currentDepthTexture;
};

} // namespace WebKit::WebGPU

#endif // ENABLE(GPU_PROCESS)
