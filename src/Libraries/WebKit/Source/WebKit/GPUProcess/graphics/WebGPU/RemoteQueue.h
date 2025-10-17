/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 18, 2024.
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
#include <cstdint>
#include <wtf/CompletionHandler.h>
#include <wtf/Ref.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/WeakRef.h>
#include <wtf/text/WTFString.h>

namespace WebCore {
class SharedMemoryHandle;
}

namespace WebCore::WebGPU {
class Queue;
}

namespace IPC {
class StreamServerConnection;
}

namespace WebKit {

namespace WebGPU {
struct ImageCopyExternalImage;
struct ImageCopyTexture;
struct ImageCopyTextureTagged;
struct ImageDataLayout;
class ObjectHeap;
}

class RemoteQueue final : public IPC::StreamMessageReceiver {
    WTF_MAKE_TZONE_ALLOCATED(RemoteQueue);
public:
    static Ref<RemoteQueue> create(WebCore::WebGPU::Queue& queue, WebGPU::ObjectHeap& objectHeap, Ref<IPC::StreamServerConnection>&& streamConnection, RemoteGPU& gpu, WebGPUIdentifier identifier)
    {
        return adoptRef(*new RemoteQueue(queue, objectHeap, WTFMove(streamConnection), gpu, identifier));
    }

    virtual ~RemoteQueue();

    // FIXME: Remove SUPPRESS_UNCOUNTED_ARG once https://github.com/llvm/llvm-project/pull/111198 lands.
    SUPPRESS_UNCOUNTED_ARG std::optional<SharedPreferencesForWebProcess> sharedPreferencesForWebProcess() const { return m_gpu->sharedPreferencesForWebProcess(); }

    void stopListeningForIPC();

private:
    friend class WebGPU::ObjectHeap;

    RemoteQueue(WebCore::WebGPU::Queue&, WebGPU::ObjectHeap&, Ref<IPC::StreamServerConnection>&&, RemoteGPU&, WebGPUIdentifier);

    RemoteQueue(const RemoteQueue&) = delete;
    RemoteQueue(RemoteQueue&&) = delete;
    RemoteQueue& operator=(const RemoteQueue&) = delete;
    RemoteQueue& operator=(RemoteQueue&&) = delete;

    WebCore::WebGPU::Queue& backing() { return m_backing; }
    Ref<WebCore::WebGPU::Queue> protectedBacking();

    Ref<WebGPU::ObjectHeap> protectedObjectHeap() const;

    void didReceiveStreamMessage(IPC::StreamServerConnection&, IPC::Decoder&) final;

    void submit(Vector<WebGPUIdentifier>&&);

    void onSubmittedWorkDone(CompletionHandler<void()>&&);

    void writeBuffer(
        WebGPUIdentifier,
        WebCore::WebGPU::Size64 bufferOffset,
        std::optional<WebCore::SharedMemoryHandle>&&,
        CompletionHandler<void(bool)>&&);

    void writeTexture(
        const WebGPU::ImageCopyTexture& destination,
        std::optional<WebCore::SharedMemoryHandle>&&,
        const WebGPU::ImageDataLayout&,
        const WebGPU::Extent3D& size,
        CompletionHandler<void(bool)>&&);

    void copyExternalImageToTexture(
        const WebGPU::ImageCopyExternalImage& source,
        const WebGPU::ImageCopyTextureTagged& destination,
        const WebGPU::Extent3D& copySize);

    void setLabel(String&&);
    void destruct();

    Ref<WebCore::WebGPU::Queue> m_backing;
    WeakRef<WebGPU::ObjectHeap> m_objectHeap;
    const Ref<IPC::StreamServerConnection> m_streamConnection;
    WeakRef<RemoteGPU> m_gpu;
    WebGPUIdentifier m_identifier;
};

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
