/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 5, 2022.
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
class RenderPipeline;
}

namespace IPC {
class StreamServerConnection;
}

namespace WebKit {

namespace WebGPU {
class ObjectHeap;
}

class RemoteRenderPipeline final : public IPC::StreamMessageReceiver {
    WTF_MAKE_TZONE_ALLOCATED(RemoteRenderPipeline);
public:
    static Ref<RemoteRenderPipeline> create(WebCore::WebGPU::RenderPipeline& renderPipeline, WebGPU::ObjectHeap& objectHeap, Ref<IPC::StreamServerConnection>&& streamConnection, RemoteGPU& gpu, WebGPUIdentifier identifier)
    {
        return adoptRef(*new RemoteRenderPipeline(renderPipeline, objectHeap, WTFMove(streamConnection), gpu, identifier));
    }

    virtual ~RemoteRenderPipeline();

    // FIXME: Remove SUPPRESS_UNCOUNTED_ARG once https://github.com/llvm/llvm-project/pull/111198 lands.
    SUPPRESS_UNCOUNTED_ARG std::optional<SharedPreferencesForWebProcess> sharedPreferencesForWebProcess() const { return m_gpu->sharedPreferencesForWebProcess(); }

    void stopListeningForIPC();

private:
    friend class WebGPU::ObjectHeap;

    RemoteRenderPipeline(WebCore::WebGPU::RenderPipeline&, WebGPU::ObjectHeap&, Ref<IPC::StreamServerConnection>&&, RemoteGPU&, WebGPUIdentifier);

    RemoteRenderPipeline(const RemoteRenderPipeline&) = delete;
    RemoteRenderPipeline(RemoteRenderPipeline&&) = delete;
    RemoteRenderPipeline& operator=(const RemoteRenderPipeline&) = delete;
    RemoteRenderPipeline& operator=(RemoteRenderPipeline&&) = delete;

    WebCore::WebGPU::RenderPipeline& backing() { return m_backing; }
    Ref<WebCore::WebGPU::RenderPipeline> protectedBacking();

    Ref<IPC::StreamServerConnection> protectedStreamConnection() const;

    Ref<WebGPU::ObjectHeap> protectedObjectHeap() const { return m_objectHeap.get(); }
    Ref<RemoteGPU> protectedGPU() const { return m_gpu.get(); }

    void didReceiveStreamMessage(IPC::StreamServerConnection&, IPC::Decoder&) final;

    void getBindGroupLayout(uint32_t index, WebGPUIdentifier);

    void setLabel(String&&);
    void destruct();

    Ref<WebCore::WebGPU::RenderPipeline> m_backing;
    WeakRef<WebGPU::ObjectHeap> m_objectHeap;
    Ref<IPC::StreamServerConnection> m_streamConnection;
    WeakRef<RemoteGPU> m_gpu;
    WebGPUIdentifier m_identifier;
};

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
