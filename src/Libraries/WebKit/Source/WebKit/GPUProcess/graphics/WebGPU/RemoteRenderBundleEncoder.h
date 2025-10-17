/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 25, 2022.
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
#include <WebCore/WebGPUIndexFormat.h>
#include <WebCore/WebGPUIntegralTypes.h>
#include <wtf/Ref.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakRef.h>
#include <wtf/text/WTFString.h>

namespace WebCore::WebGPU {
class RenderBundleEncoder;
}

namespace IPC {
class Connection;
class StreamServerConnection;
}

namespace WebKit {

class GPUConnectionToWebProcess;

namespace WebGPU {
class ObjectHeap;
struct RenderBundleDescriptor;
}

class RemoteRenderBundleEncoder final : public IPC::StreamMessageReceiver {
    WTF_MAKE_TZONE_ALLOCATED(RemoteRenderBundleEncoder);
public:
    static Ref<RemoteRenderBundleEncoder> create(GPUConnectionToWebProcess& gpuConnectionToWebProcess, RemoteGPU& gpu, WebCore::WebGPU::RenderBundleEncoder& renderBundleEncoder, WebGPU::ObjectHeap& objectHeap, Ref<IPC::StreamServerConnection>&& streamConnection, WebGPUIdentifier identifier)
    {
        return adoptRef(*new RemoteRenderBundleEncoder(gpuConnectionToWebProcess, gpu, renderBundleEncoder, objectHeap, WTFMove(streamConnection), identifier));
    }

    ~RemoteRenderBundleEncoder();

    // FIXME: Remove SUPPRESS_UNCOUNTED_ARG once https://github.com/llvm/llvm-project/pull/111198 lands.
    SUPPRESS_UNCOUNTED_ARG std::optional<SharedPreferencesForWebProcess> sharedPreferencesForWebProcess() const { return m_gpu->sharedPreferencesForWebProcess(); }

    void stopListeningForIPC();

private:
    friend class WebGPU::ObjectHeap;

    RemoteRenderBundleEncoder(GPUConnectionToWebProcess&, RemoteGPU&, WebCore::WebGPU::RenderBundleEncoder&, WebGPU::ObjectHeap&, Ref<IPC::StreamServerConnection>&&, WebGPUIdentifier);

    RemoteRenderBundleEncoder(const RemoteRenderBundleEncoder&) = delete;
    RemoteRenderBundleEncoder(RemoteRenderBundleEncoder&&) = delete;
    RemoteRenderBundleEncoder& operator=(const RemoteRenderBundleEncoder&) = delete;
    RemoteRenderBundleEncoder& operator=(RemoteRenderBundleEncoder&&) = delete;

    WebCore::WebGPU::RenderBundleEncoder& backing() { return m_backing; }
    Ref<WebCore::WebGPU::RenderBundleEncoder> protectedBacking() const;
    Ref<IPC::StreamServerConnection> protectedStreamConnection() const;
    Ref<WebGPU::ObjectHeap> protectedObjectHeap() const;
    Ref<RemoteGPU> protectedGPU() const { return m_gpu.get(); }

    RefPtr<IPC::Connection> connection() const;

    void didReceiveStreamMessage(IPC::StreamServerConnection&, IPC::Decoder&) final;

    void setPipeline(WebGPUIdentifier);

    void setIndexBuffer(WebGPUIdentifier, WebCore::WebGPU::IndexFormat, std::optional<WebCore::WebGPU::Size64> offset, std::optional<WebCore::WebGPU::Size64>);
    void setVertexBuffer(WebCore::WebGPU::Index32 slot, WebGPUIdentifier, std::optional<WebCore::WebGPU::Size64> offset, std::optional<WebCore::WebGPU::Size64>);
    void unsetVertexBuffer(WebCore::WebGPU::Index32 slot, std::optional<WebCore::WebGPU::Size64> offset, std::optional<WebCore::WebGPU::Size64>);

    void draw(WebCore::WebGPU::Size32 vertexCount, std::optional<WebCore::WebGPU::Size32> instanceCount,
        std::optional<WebCore::WebGPU::Size32> firstVertex, std::optional<WebCore::WebGPU::Size32> firstInstance);
    void drawIndexed(WebCore::WebGPU::Size32 indexCount, std::optional<WebCore::WebGPU::Size32> instanceCount,
        std::optional<WebCore::WebGPU::Size32> firstIndex,
        std::optional<WebCore::WebGPU::SignedOffset32> baseVertex,
        std::optional<WebCore::WebGPU::Size32> firstInstance);

    void drawIndirect(WebGPUIdentifier indirectBuffer, WebCore::WebGPU::Size64 indirectOffset);
    void drawIndexedIndirect(WebGPUIdentifier indirectBuffer, WebCore::WebGPU::Size64 indirectOffset);

    void setBindGroup(WebCore::WebGPU::Index32, WebGPUIdentifier,
        std::optional<Vector<WebCore::WebGPU::BufferDynamicOffset>>&& dynamicOffsets);

    void pushDebugGroup(String&& groupLabel);
    void popDebugGroup();
    void insertDebugMarker(String&& markerLabel);

    void finish(const WebGPU::RenderBundleDescriptor&, WebGPUIdentifier);

    void setLabel(String&&);
    void destruct();

    Ref<WebCore::WebGPU::RenderBundleEncoder> m_backing;
    WeakRef<WebGPU::ObjectHeap> m_objectHeap;
    Ref<IPC::StreamServerConnection> m_streamConnection;
    WebGPUIdentifier m_identifier;
    ThreadSafeWeakPtr<GPUConnectionToWebProcess> m_gpuConnectionToWebProcess;
    WeakRef<RemoteGPU> m_gpu;
};

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
