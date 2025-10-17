/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 8, 2022.
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

#include "RemoteGPURequestAdapterResponse.h"
#include "RemoteVideoFrameIdentifier.h"
#include "SharedPreferencesForWebProcess.h"
#include "StreamConnectionWorkQueue.h"
#include "StreamMessageReceiver.h"
#include "StreamServerConnection.h"
#include "WebGPUIdentifier.h"
#include "WebGPUObjectHeap.h"
#include "WebGPUSupportedFeatures.h"
#include "WebGPUSupportedLimits.h"
#include <WebCore/MediaPlayerIdentifier.h>
#include <WebCore/ProcessIdentifier.h>
#include <WebCore/RenderingResourceIdentifier.h>
#include <wtf/CompletionHandler.h>
#include <wtf/Ref.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/ThreadAssertions.h>
#include <wtf/ThreadSafeWeakPtr.h>
#include <wtf/WeakPtr.h>

namespace WebCore::WebGPU {
class GPU;
struct PresentationContextDescriptor;
}

namespace IPC {
class Connection;
class StreamServerConnection;
}

namespace WebCore {
class MediaPlayer;
class NativeImage;
class VideoFrame;
}

namespace WebKit {

class GPUConnectionToWebProcess;
class RemoteRenderingBackend;

namespace WebGPU {
class ObjectHeap;
struct RequestAdapterOptions;
}

class RemoteGPU final : public IPC::StreamMessageReceiver, public CanMakeWeakPtr<RemoteGPU> {
    WTF_MAKE_TZONE_ALLOCATED(RemoteGPU);
public:
    static Ref<RemoteGPU> create(WebGPUIdentifier identifier, GPUConnectionToWebProcess& gpuConnectionToWebProcess, RemoteRenderingBackend& renderingBackend, Ref<IPC::StreamServerConnection>&& serverConnection)
    {
        auto result = adoptRef(*new RemoteGPU(identifier, gpuConnectionToWebProcess, renderingBackend, WTFMove(serverConnection)));
        result->initialize();
        return result;
    }

    std::optional<SharedPreferencesForWebProcess> sharedPreferencesForWebProcess() const { return m_sharedPreferencesForWebProcess; }

    virtual ~RemoteGPU();

    void stopListeningForIPC();

    void paintNativeImageToImageBuffer(WebCore::NativeImage&, WebCore::RenderingResourceIdentifier);
    RefPtr<GPUConnectionToWebProcess> gpuConnectionToWebProcess() const;

private:
    friend class WebGPU::ObjectHeap;

    RemoteGPU(WebGPUIdentifier, GPUConnectionToWebProcess&, RemoteRenderingBackend&, Ref<IPC::StreamServerConnection>&&);

    RemoteGPU(const RemoteGPU&) = delete;
    RemoteGPU(RemoteGPU&&) = delete;
    RemoteGPU& operator=(const RemoteGPU&) = delete;
    RemoteGPU& operator=(RemoteGPU&&) = delete;

    RefPtr<IPC::Connection> connection() const;

    void initialize();
    IPC::StreamConnectionWorkQueue& workQueue() const { return m_workQueue; }
    Ref<IPC::StreamConnectionWorkQueue> protectedWorkQueue() const { return m_workQueue; }
    void workQueueInitialize();
    void workQueueUninitialize();

    template<typename T>
    IPC::Error send(T&& message) const
    {
        // FIXME: Remove this suppression once https://github.com/llvm/llvm-project/pull/119336 is merged.
IGNORE_CLANG_STATIC_ANALYZER_WARNINGS_BEGIN("alpha.webkit.UncountedCallArgsChecker")
        return Ref { *m_streamConnection }->send(std::forward<T>(message), m_identifier);
IGNORE_CLANG_STATIC_ANALYZER_WARNINGS_END
    }

    void didReceiveStreamMessage(IPC::StreamServerConnection&, IPC::Decoder&) final;

    void requestAdapter(const WebGPU::RequestAdapterOptions&, WebGPUIdentifier, CompletionHandler<void(std::optional<RemoteGPURequestAdapterResponse>&&)>&&);

    void createPresentationContext(const WebGPU::PresentationContextDescriptor&, WebGPUIdentifier);

    void createCompositorIntegration(WebGPUIdentifier);

    void isValid(WebGPUIdentifier, CompletionHandler<void(bool, bool)>&&);

    ThreadSafeWeakPtr<GPUConnectionToWebProcess> m_gpuConnectionToWebProcess;
    SharedPreferencesForWebProcess m_sharedPreferencesForWebProcess;
    Ref<IPC::StreamConnectionWorkQueue> m_workQueue;
    RefPtr<IPC::StreamServerConnection> m_streamConnection;
    RefPtr<WebCore::WebGPU::GPU> m_backing WTF_GUARDED_BY_CAPABILITY(workQueue());
    Ref<WebGPU::ObjectHeap> m_objectHeap WTF_GUARDED_BY_CAPABILITY(workQueue());
    const WebGPUIdentifier m_identifier;
    Ref<RemoteRenderingBackend> m_renderingBackend;
};

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
