/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 10, 2024.
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
#include <WebCore/AlphaPremultiplication.h>
#include <WebCore/RenderingResourceIdentifier.h>
#include <WebCore/WebGPUIntegralTypes.h>
#include <WebCore/WebGPUTextureUsage.h>
#include <WebCore/WebGPUXREye.h>
#include <wtf/Ref.h>
#include <wtf/WeakRef.h>
#include <wtf/text/WTFString.h>

namespace WebCore::WebGPU {
class XRBinding;
}

namespace IPC {
class StreamServerConnection;
}

namespace WebKit {

class RemoteGPU;

namespace WebGPU {
class ObjectHeap;
}

class RemoteXRBinding final : public IPC::StreamMessageReceiver {
    WTF_MAKE_TZONE_ALLOCATED(RemoteXRBinding);
public:
    static Ref<RemoteXRBinding> create(GPUConnectionToWebProcess& gpuConnectionToWebProcess, WebCore::WebGPU::XRBinding& xrBinding, WebGPU::ObjectHeap& objectHeap, RemoteGPU& gpu, Ref<IPC::StreamServerConnection>&& streamConnection, WebGPUIdentifier identifier)
    {
        return adoptRef(*new RemoteXRBinding(gpuConnectionToWebProcess, xrBinding, objectHeap, gpu, WTFMove(streamConnection), identifier));
    }

    virtual ~RemoteXRBinding();

    // FIXME: Remove SUPPRESS_UNCOUNTED_ARG once https://github.com/llvm/llvm-project/pull/111198 lands.
    SUPPRESS_UNCOUNTED_ARG std::optional<SharedPreferencesForWebProcess> sharedPreferencesForWebProcess() const { return m_gpu->sharedPreferencesForWebProcess(); }
    void stopListeningForIPC();

private:
    friend class WebGPU::ObjectHeap;

    RemoteXRBinding(GPUConnectionToWebProcess&, WebCore::WebGPU::XRBinding&, WebGPU::ObjectHeap&, RemoteGPU&, Ref<IPC::StreamServerConnection>&&, WebGPUIdentifier);

    RemoteXRBinding(const RemoteXRBinding&) = delete;
    RemoteXRBinding(RemoteXRBinding&&) = delete;
    RemoteXRBinding& operator=(const RemoteXRBinding&) = delete;
    RemoteXRBinding& operator=(RemoteXRBinding&&) = delete;

    Ref<IPC::StreamServerConnection> protectedStreamConnection();

    WebCore::WebGPU::XRBinding& backing() { return m_backing; }
    Ref<WebCore::WebGPU::XRBinding> protectedBacking();

    Ref<RemoteGPU> protectedGPU();

    void didReceiveStreamMessage(IPC::StreamServerConnection&, IPC::Decoder&) final;
    void destruct();
    void createProjectionLayer(WebCore::WebGPU::TextureFormat, std::optional<WebCore::WebGPU::TextureFormat>, WebCore::WebGPU::TextureUsageFlags, double, WebGPUIdentifier);
    void getViewSubImage(WebGPUIdentifier projectionLayerIdentifier, WebGPUIdentifier);

    Ref<WebCore::WebGPU::XRBinding> m_backing;
    WeakRef<WebGPU::ObjectHeap> m_objectHeap;
    Ref<IPC::StreamServerConnection> m_streamConnection;
    ThreadSafeWeakPtr<GPUConnectionToWebProcess> m_gpuConnectionToWebProcess;
    WebGPUIdentifier m_identifier;
    WeakRef<RemoteGPU> m_gpu;
};

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
