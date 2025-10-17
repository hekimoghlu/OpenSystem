/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 16, 2024.
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
#include <WebCore/WebGPUBindGroup.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore::WebGPU {
class ExternalTexture;
}

namespace WebKit::WebGPU {

class ConvertToBackingContext;

class RemoteBindGroupProxy final : public WebCore::WebGPU::BindGroup {
    WTF_MAKE_TZONE_ALLOCATED(RemoteBindGroupProxy);
public:
    static Ref<RemoteBindGroupProxy> create(RemoteDeviceProxy& parent, ConvertToBackingContext& convertToBackingContext, WebGPUIdentifier identifier)
    {
        return adoptRef(*new RemoteBindGroupProxy(parent, convertToBackingContext, identifier));
    }

    virtual ~RemoteBindGroupProxy();

    RemoteDeviceProxy& parent() { return m_parent; }
    RemoteGPUProxy& root() { return m_parent->root(); }

private:
    friend class DowncastConvertToBackingContext;

    RemoteBindGroupProxy(RemoteDeviceProxy&, ConvertToBackingContext&, WebGPUIdentifier);

    RemoteBindGroupProxy(const RemoteBindGroupProxy&) = delete;
    RemoteBindGroupProxy(RemoteBindGroupProxy&&) = delete;
    RemoteBindGroupProxy& operator=(const RemoteBindGroupProxy&) = delete;
    RemoteBindGroupProxy& operator=(RemoteBindGroupProxy&&) = delete;

    WebGPUIdentifier backing() const { return m_backing; }
    
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

    void setLabelInternal(const String&) final;
    bool updateExternalTextures(WebCore::WebGPU::ExternalTexture&) final;

    WebGPUIdentifier m_backing;
    Ref<ConvertToBackingContext> m_convertToBackingContext;
    Ref<RemoteDeviceProxy> m_parent;
};

} // namespace WebKit::WebGPU

#endif // ENABLE(GPU_PROCESS)
