/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 31, 2024.
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
#include <WebCore/WebGPUBuffer.h>
#include <wtf/TZoneMalloc.h>

namespace WebKit::WebGPU {

class ConvertToBackingContext;

class RemoteBufferProxy final : public WebCore::WebGPU::Buffer {
    WTF_MAKE_TZONE_ALLOCATED(RemoteBufferProxy);
public:
    static Ref<RemoteBufferProxy> create(RemoteDeviceProxy& parent, ConvertToBackingContext& convertToBackingContext, WebGPUIdentifier identifier, bool mappedAtCreation)
    {
        return adoptRef(*new RemoteBufferProxy(parent, convertToBackingContext, identifier, mappedAtCreation));
    }

    virtual ~RemoteBufferProxy();

    RemoteDeviceProxy& parent() { return m_parent; }
    RemoteGPUProxy& root() { return m_parent->root(); }

private:
    friend class DowncastConvertToBackingContext;

    RemoteBufferProxy(RemoteDeviceProxy&, ConvertToBackingContext&, WebGPUIdentifier, bool mappedAtCreation);

    RemoteBufferProxy(const RemoteBufferProxy&) = delete;
    RemoteBufferProxy(RemoteBufferProxy&&) = delete;
    RemoteBufferProxy& operator=(const RemoteBufferProxy&) = delete;
    RemoteBufferProxy& operator=(RemoteBufferProxy&&) = delete;

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
    template<typename T, typename C>
    WARN_UNUSED_RETURN std::optional<IPC::StreamClientConnection::AsyncReplyID> sendWithAsyncReply(T&& message, C&& completionHandler)
    {
        return root().protectedStreamClientConnection()->sendWithAsyncReply(WTFMove(message), completionHandler, backing());
    }

    void mapAsync(WebCore::WebGPU::MapModeFlags, WebCore::WebGPU::Size64 offset, std::optional<WebCore::WebGPU::Size64> sizeForMap, CompletionHandler<void(bool)>&&) final;
    void getMappedRange(WebCore::WebGPU::Size64 offset, std::optional<WebCore::WebGPU::Size64>, Function<void(std::span<uint8_t>)>&&) final;
    std::span<uint8_t> getBufferContents() final;
    void unmap() final;
    void copyFrom(std::span<const uint8_t>, size_t offset) final;

    void destroy() final;

    void setLabelInternal(const String&) final;

    WebGPUIdentifier m_backing;
    Ref<ConvertToBackingContext> m_convertToBackingContext;
    Ref<RemoteDeviceProxy> m_parent;
    WebCore::WebGPU::MapModeFlags m_mapModeFlags;
};

} // namespace WebKit::WebGPU

#endif // ENABLE(GPU_PROCESS)
