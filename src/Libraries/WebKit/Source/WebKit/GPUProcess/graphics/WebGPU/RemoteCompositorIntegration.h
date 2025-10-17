/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 6, 2023.
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
#include <wtf/Ref.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakRef.h>
#include <wtf/text/WTFString.h>

#if PLATFORM(COCOA)
#include <wtf/MachSendRight.h>
#include <wtf/Vector.h>
#endif

namespace WebCore {
class DestinationColorSpace;
class ImageBuffer;
}

namespace WebCore::WebGPU {
class CompositorIntegration;
}

namespace IPC {
class Connection;
class StreamServerConnection;
}

namespace WebKit {

class RemoteGPU;

namespace WebGPU {
class ObjectHeap;
}

class RemoteCompositorIntegration final : public IPC::StreamMessageReceiver {
    WTF_MAKE_TZONE_ALLOCATED(RemoteCompositorIntegration);
public:
    static Ref<RemoteCompositorIntegration> create(WebCore::WebGPU::CompositorIntegration& compositorIntegration, WebGPU::ObjectHeap& objectHeap, Ref<IPC::StreamServerConnection>&& streamConnection, RemoteGPU& gpu, WebGPUIdentifier identifier)
    {
        return adoptRef(*new RemoteCompositorIntegration(compositorIntegration, objectHeap, WTFMove(streamConnection), gpu, identifier));
    }

    virtual ~RemoteCompositorIntegration();

    // FIXME: Remove SUPPRESS_UNCOUNTED_ARG once https://github.com/llvm/llvm-project/pull/111198 lands.
    SUPPRESS_UNCOUNTED_ARG std::optional<SharedPreferencesForWebProcess> sharedPreferencesForWebProcess() const { return m_gpu->sharedPreferencesForWebProcess(); }

    void stopListeningForIPC();

private:
    friend class WebGPU::ObjectHeap;

    RemoteCompositorIntegration(WebCore::WebGPU::CompositorIntegration&, WebGPU::ObjectHeap&, Ref<IPC::StreamServerConnection>&&, RemoteGPU&, WebGPUIdentifier);

    RemoteCompositorIntegration(const RemoteCompositorIntegration&) = delete;
    RemoteCompositorIntegration(RemoteCompositorIntegration&&) = delete;
    RemoteCompositorIntegration& operator=(const RemoteCompositorIntegration&) = delete;
    RemoteCompositorIntegration& operator=(RemoteCompositorIntegration&&) = delete;

    WebCore::WebGPU::CompositorIntegration& backing() { return m_backing; }
    Ref<WebCore::WebGPU::CompositorIntegration> protectedBacking();

    Ref<IPC::StreamServerConnection> protectedStreamConnection() const;

    Ref<WebGPU::ObjectHeap> protectedObjectHeap() const { return m_objectHeap.get(); }
    Ref<RemoteGPU> protectedGPU() const { return m_gpu.get(); }

    RefPtr<IPC::Connection> connection() const;

    void didReceiveStreamMessage(IPC::StreamServerConnection&, IPC::Decoder&) final;
    void destruct();
    void paintCompositedResultsToCanvas(WebCore::RenderingResourceIdentifier, uint32_t, CompletionHandler<void()>&&);

#if PLATFORM(COCOA)
    void recreateRenderBuffers(int width, int height, WebCore::DestinationColorSpace&&, WebCore::AlphaPremultiplication, WebCore::WebGPU::TextureFormat, WebKit::WebGPUIdentifier deviceIdentifier, CompletionHandler<void(Vector<MachSendRight>&&)>&&);
#endif

    void prepareForDisplay(uint32_t frameIndex, CompletionHandler<void(bool)>&&);

    Ref<WebCore::WebGPU::CompositorIntegration> m_backing;
    WeakRef<WebGPU::ObjectHeap> m_objectHeap;
    Ref<IPC::StreamServerConnection> m_streamConnection;
    WeakRef<RemoteGPU> m_gpu;
    WebGPUIdentifier m_identifier;
};

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
