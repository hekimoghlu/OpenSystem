/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 31, 2021.
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
#include <wtf/Ref.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakRef.h>
#include <wtf/text/WTFString.h>

namespace WebCore::WebGPU {
class PresentationContext;
}

namespace IPC {
class StreamServerConnection;
}

namespace WebKit {

class GPUConnectionToWebProcess;

namespace WebGPU {
class ObjectHeap;
struct CanvasConfiguration;
}

class RemotePresentationContext final : public IPC::StreamMessageReceiver {
    WTF_MAKE_TZONE_ALLOCATED(RemotePresentationContext);
public:
    static Ref<RemotePresentationContext> create(GPUConnectionToWebProcess& gpuConnectionToWebProcess, RemoteGPU& gpu, WebCore::WebGPU::PresentationContext& presentationContext, WebGPU::ObjectHeap& objectHeap, Ref<IPC::StreamServerConnection>&& streamConnection, WebGPUIdentifier identifier)
    {
        return adoptRef(*new RemotePresentationContext(gpuConnectionToWebProcess, gpu, presentationContext, objectHeap, WTFMove(streamConnection), identifier));
    }

    virtual ~RemotePresentationContext();

    // FIXME: Remove SUPPRESS_UNCOUNTED_ARG once https://github.com/llvm/llvm-project/pull/111198 lands.
    SUPPRESS_UNCOUNTED_ARG std::optional<SharedPreferencesForWebProcess> sharedPreferencesForWebProcess() const { return m_gpu->sharedPreferencesForWebProcess(); }

    void stopListeningForIPC();

private:
    friend class WebGPU::ObjectHeap;

    RemotePresentationContext(GPUConnectionToWebProcess&, RemoteGPU&, WebCore::WebGPU::PresentationContext&, WebGPU::ObjectHeap&, Ref<IPC::StreamServerConnection>&&, WebGPUIdentifier);

    RemotePresentationContext(const RemotePresentationContext&) = delete;
    RemotePresentationContext(RemotePresentationContext&&) = delete;
    RemotePresentationContext& operator=(const RemotePresentationContext&) = delete;
    RemotePresentationContext& operator=(RemotePresentationContext&&) = delete;

    WebCore::WebGPU::PresentationContext& backing() { return m_backing; }
    Ref<WebCore::WebGPU::PresentationContext> protectedBacking();
    Ref<IPC::StreamServerConnection> protectedStreamConnection() const;
    Ref<WebGPU::ObjectHeap> protectedObjectHeap() const;
    Ref<RemoteGPU> protectedGPU() const { return m_gpu.get(); }

    void didReceiveStreamMessage(IPC::StreamServerConnection&, IPC::Decoder&) final;

    void configure(const WebGPU::CanvasConfiguration&);
    void unconfigure();
    void present(uint32_t frameIndex);

    void getCurrentTexture(WebGPUIdentifier, uint32_t frameIndex);

    Ref<WebCore::WebGPU::PresentationContext> m_backing;
    WeakRef<WebGPU::ObjectHeap> m_objectHeap;
    Ref<IPC::StreamServerConnection> m_streamConnection;
    WebGPUIdentifier m_identifier;
    ThreadSafeWeakPtr<GPUConnectionToWebProcess> m_gpuConnectionToWebProcess;
    WeakRef<RemoteGPU> m_gpu;
};

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
