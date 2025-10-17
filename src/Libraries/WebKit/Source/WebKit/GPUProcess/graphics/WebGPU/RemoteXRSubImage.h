/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 19, 2024.
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
class XRSubImage;
}

namespace IPC {
class Connection;
class StreamServerConnection;
}

namespace WebKit {

class GPUConnectionToWebProcess;
class RemoteGPU;

namespace WebGPU {
class ObjectHeap;
}

class RemoteXRSubImage final : public IPC::StreamMessageReceiver {
    WTF_MAKE_TZONE_ALLOCATED(RemoteXRSubImage);
public:
    static Ref<RemoteXRSubImage> create(GPUConnectionToWebProcess& gpuConnectionToWebProcess, WebCore::WebGPU::XRSubImage& xrSubImage, WebGPU::ObjectHeap& objectHeap, Ref<IPC::StreamServerConnection>&& streamConnection, RemoteGPU& gpu, WebGPUIdentifier identifier)
    {
        return adoptRef(*new RemoteXRSubImage(gpuConnectionToWebProcess, xrSubImage, objectHeap, WTFMove(streamConnection), gpu, identifier));
    }

    virtual ~RemoteXRSubImage();

    // FIXME: Remove SUPPRESS_UNCOUNTED_ARG once https://github.com/llvm/llvm-project/pull/111198 lands.
    SUPPRESS_UNCOUNTED_ARG std::optional<SharedPreferencesForWebProcess> sharedPreferencesForWebProcess() const { return m_gpu->sharedPreferencesForWebProcess(); }
    void stopListeningForIPC();

private:
    friend class WebGPU::ObjectHeap;

    RemoteXRSubImage(GPUConnectionToWebProcess&, WebCore::WebGPU::XRSubImage&, WebGPU::ObjectHeap&, Ref<IPC::StreamServerConnection>&&, RemoteGPU&, WebGPUIdentifier);

    RemoteXRSubImage(const RemoteXRSubImage&) = delete;
    RemoteXRSubImage(RemoteXRSubImage&&) = delete;
    RemoteXRSubImage& operator=(const RemoteXRSubImage&) = delete;
    RemoteXRSubImage& operator=(RemoteXRSubImage&&) = delete;

    WebCore::WebGPU::XRSubImage& backing() { return m_backing; }
    Ref<WebCore::WebGPU::XRSubImage> protectedBacking();

    Ref<IPC::StreamServerConnection> protectedStreamConnection();
    Ref<RemoteGPU> protectedGPU() const;

    RefPtr<IPC::Connection> connection() const;

    void didReceiveStreamMessage(IPC::StreamServerConnection&, IPC::Decoder&) final;
    void destruct();
    void getColorTexture(WebGPUIdentifier);
    void getDepthTexture(WebGPUIdentifier);

    Ref<WebCore::WebGPU::XRSubImage> m_backing;
    WeakRef<WebGPU::ObjectHeap> m_objectHeap;
    Ref<IPC::StreamServerConnection> m_streamConnection;
    ThreadSafeWeakPtr<GPUConnectionToWebProcess> m_gpuConnectionToWebProcess;
    WebGPUIdentifier m_identifier;
    WeakRef<RemoteGPU> m_gpu;
};

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
