/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 23, 2025.
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
#include <WebCore/WebGPUAdapter.h>
#include <wtf/TZoneMalloc.h>

namespace WebKit {
class RemoteGPUProxy;
}

namespace WebKit::WebGPU {

class ConvertToBackingContext;

class RemoteAdapterProxy final : public WebCore::WebGPU::Adapter {
    WTF_MAKE_TZONE_ALLOCATED(RemoteAdapterProxy);
public:
    static Ref<RemoteAdapterProxy> create(String&& name, WebCore::WebGPU::SupportedFeatures& features, WebCore::WebGPU::SupportedLimits& limits, bool isFallbackAdapter, bool xrCompatible, RemoteGPUProxy& parent, ConvertToBackingContext& convertToBackingContext, WebGPUIdentifier identifier)
    {
        return adoptRef(*new RemoteAdapterProxy(WTFMove(name), features, limits, isFallbackAdapter, xrCompatible, parent, convertToBackingContext, identifier));
    }

    virtual ~RemoteAdapterProxy();

    RemoteGPUProxy& parent() { return m_parent; }
    RemoteGPUProxy& root() { return m_parent->root(); }

private:
    friend class DowncastConvertToBackingContext;

    RemoteAdapterProxy(String&& name, WebCore::WebGPU::SupportedFeatures&, WebCore::WebGPU::SupportedLimits&, bool isFallbackAdapter, bool xrCompatible, RemoteGPUProxy&, ConvertToBackingContext&, WebGPUIdentifier);

    RemoteAdapterProxy(const RemoteAdapterProxy&) = delete;
    RemoteAdapterProxy(RemoteAdapterProxy&&) = delete;
    RemoteAdapterProxy& operator=(const RemoteAdapterProxy&) = delete;
    RemoteAdapterProxy& operator=(RemoteAdapterProxy&&) = delete;

    WebGPUIdentifier backing() const { return m_backing; }
    bool xrCompatible() final;
    
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

    void requestDevice(const WebCore::WebGPU::DeviceDescriptor&, CompletionHandler<void(RefPtr<WebCore::WebGPU::Device>&&)>&&) final;

    WebGPUIdentifier m_backing;
    Ref<ConvertToBackingContext> m_convertToBackingContext;
    Ref<RemoteGPUProxy> m_parent;
    bool m_xrCompatible { false };
};

} // namespace WebKit::WebGPU

#endif // ENABLE(GPU_PROCESS)
