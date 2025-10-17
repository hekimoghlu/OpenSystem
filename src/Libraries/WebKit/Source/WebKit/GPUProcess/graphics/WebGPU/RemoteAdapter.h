/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 19, 2025.
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
#include <wtf/CompletionHandler.h>
#include <wtf/Ref.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/ThreadSafeWeakPtr.h>
#include <wtf/WeakRef.h>

namespace WebCore::WebGPU {
class Adapter;
}

namespace IPC {
class StreamServerConnection;
}

namespace WebKit {

namespace WebGPU {
struct DeviceDescriptor;
class ObjectHeap;
struct SupportedFeatures;
struct SupportedLimits;
}

class RemoteAdapter final : public IPC::StreamMessageReceiver {
    WTF_MAKE_TZONE_ALLOCATED(RemoteAdapter);
public:
    static Ref<RemoteAdapter> create(GPUConnectionToWebProcess& gpuConnectionToWebProcess, RemoteGPU& gpu, WebCore::WebGPU::Adapter& adapter, WebGPU::ObjectHeap& objectHeap, Ref<IPC::StreamServerConnection>&& streamConnection, WebGPUIdentifier identifier)
    {
        return adoptRef(*new RemoteAdapter(gpuConnectionToWebProcess, gpu, adapter, objectHeap, WTFMove(streamConnection), identifier));
    }

    virtual ~RemoteAdapter();

    // FIXME: Remove SUPPRESS_UNCOUNTED_ARG once https://github.com/llvm/llvm-project/pull/111198 lands.
    SUPPRESS_UNCOUNTED_ARG std::optional<SharedPreferencesForWebProcess> sharedPreferencesForWebProcess() const { return m_gpu->sharedPreferencesForWebProcess(); }

    void stopListeningForIPC();

private:
    friend class WebGPU::ObjectHeap;

    RemoteAdapter(GPUConnectionToWebProcess&, RemoteGPU&, WebCore::WebGPU::Adapter&, WebGPU::ObjectHeap&, Ref<IPC::StreamServerConnection>&&, WebGPUIdentifier);

    RemoteAdapter(const RemoteAdapter&) = delete;
    RemoteAdapter(RemoteAdapter&&) = delete;
    RemoteAdapter& operator=(const RemoteAdapter&) = delete;
    RemoteAdapter& operator=(RemoteAdapter&&) = delete;

    WebCore::WebGPU::Adapter& backing() { return m_backing; }
    Ref<WebGPU::ObjectHeap> protectedObjectHeap() const { return m_objectHeap.get(); }
    Ref<RemoteGPU> protectedGPU() const { return m_gpu.get(); }

    void didReceiveStreamMessage(IPC::StreamServerConnection&, IPC::Decoder&) final;

    void requestDevice(const WebGPU::DeviceDescriptor&, WebGPUIdentifier, WebGPUIdentifier queueIdentifier, CompletionHandler<void(WebGPU::SupportedFeatures&&, WebGPU::SupportedLimits&&)>&&);
    void destruct();

    Ref<WebCore::WebGPU::Adapter> m_backing;
    WeakRef<WebGPU::ObjectHeap> m_objectHeap;
    const Ref<IPC::StreamServerConnection> m_streamConnection;
    ThreadSafeWeakPtr<GPUConnectionToWebProcess> m_gpuConnectionToWebProcess;
    WeakRef<RemoteGPU> m_gpu;
    WebGPUIdentifier m_identifier;
};

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
