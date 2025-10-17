/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 10, 2025.
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
#include <WebCore/WebGPUIntegralTypes.h>
#include <WebCore/WebGPUPresentationContext.h>
#include <array>
#include <wtf/TZoneMalloc.h>

namespace WebCore {
class NativeImage;
}

namespace WebKit::WebGPU {

class ConvertToBackingContext;
class RemoteTextureProxy;

class RemotePresentationContextProxy final : public WebCore::WebGPU::PresentationContext {
    WTF_MAKE_TZONE_ALLOCATED(RemotePresentationContextProxy);
public:
    static Ref<RemotePresentationContextProxy> create(RemoteGPUProxy& parent, ConvertToBackingContext& convertToBackingContext, WebGPUIdentifier identifier)
    {
        return adoptRef(*new RemotePresentationContextProxy(parent, convertToBackingContext, identifier));
    }

    virtual ~RemotePresentationContextProxy();

    RemoteGPUProxy& parent() { return m_parent; }
    RemoteGPUProxy& root() { return m_parent->root(); }
    Ref<RemoteGPUProxy> protectedRoot() { return m_parent->root(); }

    void present(uint32_t frameIndex, bool = false) final;

private:
    friend class DowncastConvertToBackingContext;

    RemotePresentationContextProxy(RemoteGPUProxy&, ConvertToBackingContext&, WebGPUIdentifier);

    RemotePresentationContextProxy(const RemotePresentationContextProxy&) = delete;
    RemotePresentationContextProxy(RemotePresentationContextProxy&&) = delete;
    RemotePresentationContextProxy& operator=(const RemotePresentationContextProxy&) = delete;
    RemotePresentationContextProxy& operator=(RemotePresentationContextProxy&&) = delete;

    WebGPUIdentifier backing() const { return m_backing; }
    Ref<ConvertToBackingContext> protectedConvertToBackingContext() const;

    RefPtr<WebCore::NativeImage> getMetalTextureAsNativeImage(uint32_t, bool& isIOSurfaceSupportedFormat) final;

    template<typename T>
    WARN_UNUSED_RETURN IPC::Error send(T&& message)
    {
        return root().protectedStreamClientConnection()->send(WTFMove(message), backing());
    }

    bool configure(const WebCore::WebGPU::CanvasConfiguration&) final;
    void unconfigure() final;

    RefPtr<WebCore::WebGPU::Texture> getCurrentTexture(uint32_t) final;

    WebGPUIdentifier m_backing;
    Ref<ConvertToBackingContext> m_convertToBackingContext;
    Ref<RemoteGPUProxy> m_parent;
    static constexpr size_t textureCount = 3;
    std::array<RefPtr<RemoteTextureProxy>, textureCount> m_currentTexture;
};

} // namespace WebKit::WebGPU

#endif // ENABLE(GPU_PROCESS)
