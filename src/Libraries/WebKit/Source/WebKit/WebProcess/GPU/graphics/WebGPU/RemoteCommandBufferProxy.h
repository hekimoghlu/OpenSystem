/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 1, 2025.
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
#include <WebCore/WebGPUCommandBuffer.h>
#include <wtf/TZoneMalloc.h>

namespace WebKit::WebGPU {

class ConvertToBackingContext;

class RemoteCommandBufferProxy final : public WebCore::WebGPU::CommandBuffer {
    WTF_MAKE_TZONE_ALLOCATED(RemoteCommandBufferProxy);
public:
    static Ref<RemoteCommandBufferProxy> create(RemoteGPUProxy& root, ConvertToBackingContext& convertToBackingContext, WebGPUIdentifier identifier)
    {
        return adoptRef(*new RemoteCommandBufferProxy(root, convertToBackingContext, identifier));
    }

    virtual ~RemoteCommandBufferProxy();

    RemoteGPUProxy& root() { return m_root; }

private:
    friend class DowncastConvertToBackingContext;

    RemoteCommandBufferProxy(RemoteGPUProxy&, ConvertToBackingContext&, WebGPUIdentifier);

    RemoteCommandBufferProxy(const RemoteCommandBufferProxy&) = delete;
    RemoteCommandBufferProxy(RemoteCommandBufferProxy&&) = delete;
    RemoteCommandBufferProxy& operator=(const RemoteCommandBufferProxy&) = delete;
    RemoteCommandBufferProxy& operator=(RemoteCommandBufferProxy&&) = delete;

    WebGPUIdentifier backing() const { return m_backing; }
    
    template<typename T>
    WARN_UNUSED_RETURN IPC::Error send(T&& message)
    {
        return root().protectedStreamClientConnection()->send(WTFMove(message), backing());
    }

    void setLabelInternal(const String&) final;

    WebGPUIdentifier m_backing;
    Ref<ConvertToBackingContext> m_convertToBackingContext;
    Ref<RemoteGPUProxy> m_root;
};

} // namespace WebKit::WebGPU

#endif // ENABLE(GPU_PROCESS)
