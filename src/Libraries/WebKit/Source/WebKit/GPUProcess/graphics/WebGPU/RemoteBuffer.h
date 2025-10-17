/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 10, 2021.
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

#include "RemoteGPU.h"
#include "StreamMessageReceiver.h"
#include "WebGPUIdentifier.h"
#include <WebCore/WebGPUBuffer.h>
#include <WebCore/WebGPUIntegralTypes.h>
#include <WebCore/WebGPUMapMode.h>
#include <wtf/CompletionHandler.h>
#include <wtf/Ref.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/WeakRef.h>
#include <wtf/text/WTFString.h>

namespace WebCore {
class SharedMemoryHandle;
}

namespace WebCore::WebGPU {
class Buffer;
}

namespace IPC {
class StreamServerConnection;
}

namespace WebKit {

namespace WebGPU {
class ObjectHeap;
}

class RemoteBuffer final : public IPC::StreamMessageReceiver {
    WTF_MAKE_TZONE_ALLOCATED(RemoteBuffer);
public:
    static Ref<RemoteBuffer> create(WebCore::WebGPU::Buffer& buffer, WebGPU::ObjectHeap& objectHeap, Ref<IPC::StreamServerConnection>&& streamConnection, RemoteGPU& gpu, bool mappedAtCreation, WebGPUIdentifier identifier)
    {
        return adoptRef(*new RemoteBuffer(buffer, objectHeap, WTFMove(streamConnection), gpu, mappedAtCreation, identifier));
    }

    virtual ~RemoteBuffer();

    // FIXME: Remove SUPPRESS_UNCOUNTED_ARG once https://github.com/llvm/llvm-project/pull/111198 lands.
    SUPPRESS_UNCOUNTED_ARG std::optional<SharedPreferencesForWebProcess> sharedPreferencesForWebProcess() const { return m_gpu->sharedPreferencesForWebProcess(); }

    void stopListeningForIPC();

private:
    friend class WebGPU::ObjectHeap;

    RemoteBuffer(WebCore::WebGPU::Buffer&, WebGPU::ObjectHeap&, Ref<IPC::StreamServerConnection>&&, RemoteGPU&, bool mappedAtCreation, WebGPUIdentifier);

    RemoteBuffer(const RemoteBuffer&) = delete;
    RemoteBuffer(RemoteBuffer&&) = delete;
    RemoteBuffer& operator=(const RemoteBuffer&) = delete;
    RemoteBuffer& operator=(RemoteBuffer&&) = delete;

    WebCore::WebGPU::Buffer& backing() { return m_backing; }
    Ref<WebCore::WebGPU::Buffer> protectedBacking();

    Ref<IPC::StreamServerConnection> protectedStreamConnection() const;

    void didReceiveStreamMessage(IPC::StreamServerConnection&, IPC::Decoder&) final;

    void getMappedRange(WebCore::WebGPU::Size64 offset, std::optional<WebCore::WebGPU::Size64> sizeForMap, CompletionHandler<void(std::optional<Vector<uint8_t>>&&)>&&);
    void mapAsync(WebCore::WebGPU::MapModeFlags, WebCore::WebGPU::Size64 offset, std::optional<WebCore::WebGPU::Size64> sizeForMap, CompletionHandler<void(bool)>&&);
    void copy(std::optional<WebCore::SharedMemoryHandle>&&, size_t offset, CompletionHandler<void(bool)>&&);
    void unmap();

    void destroy();
    void destruct();

    void setLabel(String&&);

    Ref<WebCore::WebGPU::Buffer> m_backing;
    WeakRef<WebGPU::ObjectHeap> m_objectHeap;
    Ref<IPC::StreamServerConnection> m_streamConnection;
    WeakRef<RemoteGPU> m_gpu;
    WebGPUIdentifier m_identifier;
    bool m_isMapped { false };
    WebCore::WebGPU::MapModeFlags m_mapModeFlags;
};

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
