/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 16, 2024.
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
#include "WebGPUIdentifier.h"
#include <WebCore/WebGPUIntegralTypes.h>
#include <WebCore/WebGPUTexture.h>
#include <WebCore/WebGPUTextureDimension.h>
#include <WebCore/WebGPUTextureFormat.h>
#include <WebCore/WebGPUTextureViewDescriptor.h>
#include <wtf/TZoneMalloc.h>

namespace WebKit::WebGPU {

class ConvertToBackingContext;

class RemoteTextureProxy final : public WebCore::WebGPU::Texture {
    WTF_MAKE_TZONE_ALLOCATED(RemoteTextureProxy);
public:
    static Ref<RemoteTextureProxy> create(Ref<RemoteGPUProxy>&& root, ConvertToBackingContext& convertToBackingContext, WebGPUIdentifier identifier, bool isCanvasBacking = false)
    {
        return adoptRef(*new RemoteTextureProxy(WTFMove(root), convertToBackingContext, identifier, isCanvasBacking));
    }

    virtual ~RemoteTextureProxy();

    RemoteGPUProxy& root() { return m_root; }
    void undestroy() final;

private:
    friend class DowncastConvertToBackingContext;

    RemoteTextureProxy(Ref<RemoteGPUProxy>&&, ConvertToBackingContext&, WebGPUIdentifier, bool isCanvasBacking);

    RemoteTextureProxy(const RemoteTextureProxy&) = delete;
    RemoteTextureProxy(RemoteTextureProxy&&) = delete;
    RemoteTextureProxy& operator=(const RemoteTextureProxy&) = delete;
    RemoteTextureProxy& operator=(RemoteTextureProxy&&) = delete;

    WebGPUIdentifier backing() const { return m_backing; }
    
    template<typename T>
    WARN_UNUSED_RETURN IPC::Error send(T&& message)
    {
        return root().protectedStreamClientConnection()->send(WTFMove(message), backing());
    }

    RefPtr<WebCore::WebGPU::TextureView> createView(const std::optional<WebCore::WebGPU::TextureViewDescriptor>&) final;

    void destroy() final;
    void setLabelInternal(const String&) final;

    WebGPUIdentifier m_backing;
    Ref<ConvertToBackingContext> m_convertToBackingContext;
    Ref<RemoteGPUProxy> m_root;

    RefPtr<WebCore::WebGPU::TextureView> m_lastCreatedView;
    std::optional<WebCore::WebGPU::TextureViewDescriptor> m_lastCreatedViewDescriptor;
    bool m_isCanvasBacking { false };
};

} // namespace WebKit::WebGPU

#endif // ENABLE(GPU_PROCESS)
