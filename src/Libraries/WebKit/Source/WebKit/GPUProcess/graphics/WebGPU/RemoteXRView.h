/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 20, 2022.
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
class XRView;
}

namespace IPC {
class StreamServerConnection;
}

namespace WebKit {

class RemoteGPU;

namespace WebGPU {
class ObjectHeap;
}

class RemoteXRView final : public IPC::StreamMessageReceiver {
    WTF_MAKE_TZONE_ALLOCATED(RemoteXRView);
public:
    static Ref<RemoteXRView> create(WebCore::WebGPU::XRView& xrView, WebGPU::ObjectHeap& objectHeap, RemoteGPU& gpu, Ref<IPC::StreamServerConnection>&& streamConnection, WebGPUIdentifier identifier)
    {
        return adoptRef(*new RemoteXRView(xrView, objectHeap, gpu, WTFMove(streamConnection), identifier));
    }

    virtual ~RemoteXRView();

    // FIXME: Remove SUPPRESS_UNCOUNTED_ARG once https://github.com/llvm/llvm-project/pull/111198 lands.
    SUPPRESS_UNCOUNTED_ARG std::optional<SharedPreferencesForWebProcess> sharedPreferencesForWebProcess() const { return m_gpu->sharedPreferencesForWebProcess(); }
    void stopListeningForIPC();

private:
    friend class WebGPU::ObjectHeap;

    RemoteXRView(WebCore::WebGPU::XRView&, WebGPU::ObjectHeap&, RemoteGPU&, Ref<IPC::StreamServerConnection>&&, WebGPUIdentifier);

    RemoteXRView(const RemoteXRView&) = delete;
    RemoteXRView(RemoteXRView&&) = delete;
    RemoteXRView& operator=(const RemoteXRView&) = delete;
    RemoteXRView& operator=(RemoteXRView&&) = delete;

    Ref<IPC::StreamServerConnection> protectedStreamConnection();

    WebCore::WebGPU::XRView& backing() { return m_backing; }

    void didReceiveStreamMessage(IPC::StreamServerConnection&, IPC::Decoder&) final;
    void destruct();

    Ref<WebCore::WebGPU::XRView> m_backing;
    WeakRef<WebGPU::ObjectHeap> m_objectHeap;
    Ref<IPC::StreamServerConnection> m_streamConnection;
    WebGPUIdentifier m_identifier;
    WeakRef<RemoteGPU> m_gpu;
};

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
