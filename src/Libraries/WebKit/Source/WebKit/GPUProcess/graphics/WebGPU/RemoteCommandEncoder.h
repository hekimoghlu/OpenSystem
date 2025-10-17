/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 10, 2023.
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
#include "WebGPUExtent3D.h"
#include "WebGPUIdentifier.h"
#include <WebCore/WebGPUIntegralTypes.h>
#include <wtf/Ref.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakRef.h>
#include <wtf/text/WTFString.h>

namespace WebCore::WebGPU {
class CommandEncoder;
}

namespace IPC {
class Connection;
class StreamServerConnection;
}

namespace WebKit {

class GPUConnectionToWebProcess;

namespace WebGPU {
struct CommandBufferDescriptor;
struct ComputePassDescriptor;
struct ImageCopyBuffer;
struct ImageCopyTexture;
class ObjectHeap;
struct RenderPassDescriptor;
}

class RemoteCommandEncoder final : public IPC::StreamMessageReceiver {
    WTF_MAKE_TZONE_ALLOCATED(RemoteCommandEncoder);
public:
    static Ref<RemoteCommandEncoder> create(GPUConnectionToWebProcess& gpuConnectionToWebProcess, RemoteGPU& gpu, WebCore::WebGPU::CommandEncoder& commandEncoder, WebGPU::ObjectHeap& objectHeap, Ref<IPC::StreamServerConnection>&& streamConnection, WebGPUIdentifier identifier)
    {
        return adoptRef(*new RemoteCommandEncoder(gpuConnectionToWebProcess, gpu, commandEncoder, objectHeap, WTFMove(streamConnection), identifier));
    }

    virtual ~RemoteCommandEncoder();

    // FIXME: Remove SUPPRESS_UNCOUNTED_ARG once https://github.com/llvm/llvm-project/pull/111198 lands.
    SUPPRESS_UNCOUNTED_ARG std::optional<SharedPreferencesForWebProcess> sharedPreferencesForWebProcess() const { return m_gpu->sharedPreferencesForWebProcess(); }

    void stopListeningForIPC();

private:
    friend class WebGPU::ObjectHeap;

    RemoteCommandEncoder(GPUConnectionToWebProcess&, RemoteGPU&, WebCore::WebGPU::CommandEncoder&, WebGPU::ObjectHeap&, Ref<IPC::StreamServerConnection>&&, WebGPUIdentifier);

    RemoteCommandEncoder(const RemoteCommandEncoder&) = delete;
    RemoteCommandEncoder(RemoteCommandEncoder&&) = delete;
    RemoteCommandEncoder& operator=(const RemoteCommandEncoder&) = delete;
    RemoteCommandEncoder& operator=(RemoteCommandEncoder&&) = delete;

    WebCore::WebGPU::CommandEncoder& backing() { return m_backing; }
    Ref<WebCore::WebGPU::CommandEncoder> protectedBacking();

    Ref<WebGPU::ObjectHeap> protectedObjectHeap() const { return m_objectHeap.get(); }
    Ref<RemoteGPU> protectedGPU() const { return m_gpu.get(); }

    RefPtr<IPC::Connection> connection() const;

    void didReceiveStreamMessage(IPC::StreamServerConnection&, IPC::Decoder&) final;

    void beginRenderPass(const WebGPU::RenderPassDescriptor&, WebGPUIdentifier);
    void beginComputePass(const std::optional<WebGPU::ComputePassDescriptor>&, WebGPUIdentifier);

    void copyBufferToBuffer(
        WebGPUIdentifier source,
        WebCore::WebGPU::Size64 sourceOffset,
        WebGPUIdentifier destination,
        WebCore::WebGPU::Size64 destinationOffset,
        WebCore::WebGPU::Size64);

    void copyBufferToTexture(
        const WebGPU::ImageCopyBuffer& source,
        const WebGPU::ImageCopyTexture& destination,
        const WebGPU::Extent3D& copySize);

    void copyTextureToBuffer(
        const WebGPU::ImageCopyTexture& source,
        const WebGPU::ImageCopyBuffer& destination,
        const WebGPU::Extent3D& copySize);

    void copyTextureToTexture(
        const WebGPU::ImageCopyTexture& source,
        const WebGPU::ImageCopyTexture& destination,
        const WebGPU::Extent3D& copySize);

    void clearBuffer(
        WebGPUIdentifier buffer,
        WebCore::WebGPU::Size64 offset = 0,
        std::optional<WebCore::WebGPU::Size64> = std::nullopt);

    void pushDebugGroup(String&& groupLabel);
    void popDebugGroup();
    void insertDebugMarker(String&& markerLabel);

    void writeTimestamp(WebGPUIdentifier, WebCore::WebGPU::Size32 queryIndex);

    void resolveQuerySet(
        WebGPUIdentifier,
        WebCore::WebGPU::Size32 firstQuery,
        WebCore::WebGPU::Size32 queryCount,
        WebGPUIdentifier destination,
        WebCore::WebGPU::Size64 destinationOffset);

    void finish(const WebGPU::CommandBufferDescriptor&, WebGPUIdentifier);

    void setLabel(String&&);
    void destruct();

    Ref<WebCore::WebGPU::CommandEncoder> m_backing;
    WeakRef<WebGPU::ObjectHeap> m_objectHeap;
    const Ref<IPC::StreamServerConnection> m_streamConnection;
    WebGPUIdentifier m_identifier;
    ThreadSafeWeakPtr<GPUConnectionToWebProcess> m_gpuConnectionToWebProcess;
    WeakRef<RemoteGPU> m_gpu;
};

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
