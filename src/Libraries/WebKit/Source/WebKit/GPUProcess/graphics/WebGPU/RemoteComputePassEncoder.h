/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 4, 2023.
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
#include <WebCore/WebGPUIntegralTypes.h>
#include <wtf/Ref.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakRef.h>
#include <wtf/text/WTFString.h>

namespace WebCore::WebGPU {
class ComputePassEncoder;
}

namespace IPC {
class StreamServerConnection;
}

namespace WebKit {

namespace WebGPU {
class ObjectHeap;
}

class RemoteComputePassEncoder final : public IPC::StreamMessageReceiver {
    WTF_MAKE_TZONE_ALLOCATED(RemoteComputePassEncoder);
public:
    static Ref<RemoteComputePassEncoder> create(WebCore::WebGPU::ComputePassEncoder& computePassEncoder, WebGPU::ObjectHeap& objectHeap, Ref<IPC::StreamServerConnection>&& streamConnection, RemoteGPU& gpu, WebGPUIdentifier identifier)
    {
        return adoptRef(*new RemoteComputePassEncoder(computePassEncoder, objectHeap, WTFMove(streamConnection), gpu, identifier));
    }

    virtual ~RemoteComputePassEncoder();

    // FIXME: Remove SUPPRESS_UNCOUNTED_ARG once https://github.com/llvm/llvm-project/pull/111198 lands.
    SUPPRESS_UNCOUNTED_ARG std::optional<SharedPreferencesForWebProcess> sharedPreferencesForWebProcess() const { return m_gpu->sharedPreferencesForWebProcess(); }

    void stopListeningForIPC();

private:
    friend class WebGPU::ObjectHeap;

    RemoteComputePassEncoder(WebCore::WebGPU::ComputePassEncoder&, WebGPU::ObjectHeap&, Ref<IPC::StreamServerConnection>&&, RemoteGPU&, WebGPUIdentifier);

    RemoteComputePassEncoder(const RemoteComputePassEncoder&) = delete;
    RemoteComputePassEncoder(RemoteComputePassEncoder&&) = delete;
    RemoteComputePassEncoder& operator=(const RemoteComputePassEncoder&) = delete;
    RemoteComputePassEncoder& operator=(RemoteComputePassEncoder&&) = delete;

    WebCore::WebGPU::ComputePassEncoder& backing() { return m_backing; }
    Ref<WebCore::WebGPU::ComputePassEncoder> protectedBacking();

    Ref<IPC::StreamServerConnection> protectedStreamConnection() const;
    Ref<WebGPU::ObjectHeap> protectedObjectHeap() const;

    void didReceiveStreamMessage(IPC::StreamServerConnection&, IPC::Decoder&) final;

    void setPipeline(WebGPUIdentifier);
    void dispatch(WebCore::WebGPU::Size32 workgroupCountX, WebCore::WebGPU::Size32 workgroupCountY = 1, WebCore::WebGPU::Size32 workgroupCountZ = 1);
    void dispatchIndirect(WebGPUIdentifier indirectBuffer, WebCore::WebGPU::Size64 indirectOffset);

    void end();

    void setBindGroup(WebCore::WebGPU::Index32, WebGPUIdentifier,
        std::optional<Vector<WebCore::WebGPU::BufferDynamicOffset>>&&);

    void pushDebugGroup(String&& groupLabel);
    void popDebugGroup();
    void insertDebugMarker(String&& markerLabel);

    void setLabel(String&&);
    void destruct();

    Ref<WebCore::WebGPU::ComputePassEncoder> m_backing;
    WeakRef<WebGPU::ObjectHeap> m_objectHeap;
    Ref<IPC::StreamServerConnection> m_streamConnection;
    WeakRef<RemoteGPU> m_gpu;
    WebGPUIdentifier m_identifier;
};

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
