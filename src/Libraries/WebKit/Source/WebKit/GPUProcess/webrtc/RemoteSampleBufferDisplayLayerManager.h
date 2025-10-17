/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 19, 2023.
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

#if PLATFORM(COCOA) && ENABLE(GPU_PROCESS) && ENABLE(MEDIA_STREAM)

#include "Connection.h"
#include "LayerHostingContext.h"
#include "SampleBufferDisplayLayerIdentifier.h"
#include "SharedPreferencesForWebProcess.h"
#include "WorkQueueMessageReceiver.h"
#include <WebCore/FloatRect.h>
#include <WebCore/IntSize.h>
#include <wtf/HashMap.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakRef.h>
#include <wtf/WorkQueue.h>

namespace IPC {
class Decoder;
}

namespace WebCore {
class IntSize;
}

namespace WebKit {

class GPUConnectionToWebProcess;
class RemoteSampleBufferDisplayLayer;

class RemoteSampleBufferDisplayLayerManager final : public IPC::WorkQueueMessageReceiver {
    WTF_MAKE_TZONE_ALLOCATED(RemoteSampleBufferDisplayLayerManager);
public:
    static Ref<RemoteSampleBufferDisplayLayerManager> create(GPUConnectionToWebProcess& connection, SharedPreferencesForWebProcess& sharedPreferencesForWebProcess)
    {
        auto instance = adoptRef(*new RemoteSampleBufferDisplayLayerManager(connection, sharedPreferencesForWebProcess));
        instance->startListeningForIPC();
        return instance;
    }
    ~RemoteSampleBufferDisplayLayerManager();

    void ref() const final { IPC::WorkQueueMessageReceiver::ref(); }
    void deref() const final { IPC::WorkQueueMessageReceiver::deref(); }

    void close();

    bool allowsExitUnderMemoryPressure() const;
    void updateSampleBufferDisplayLayerBoundsAndPosition(SampleBufferDisplayLayerIdentifier, WebCore::FloatRect, std::optional<MachSendRight>&&);
    std::optional<SharedPreferencesForWebProcess> sharedPreferencesForWebProcess() const { return m_sharedPreferencesForWebProcess; }
    void updateSharedPreferencesForWebProcess(SharedPreferencesForWebProcess);

private:
    explicit RemoteSampleBufferDisplayLayerManager(GPUConnectionToWebProcess&, SharedPreferencesForWebProcess&);
    void startListeningForIPC();

    // IPC::WorkQueueMessageReceiver overrides.
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;

    bool dispatchMessage(IPC::Connection&, IPC::Decoder&);

    using LayerCreationCallback = CompletionHandler<void(std::optional<LayerHostingContextID>)>&&;
    void createLayer(SampleBufferDisplayLayerIdentifier, bool hideRootLayer, WebCore::IntSize, bool shouldMaintainAspectRatio, bool canShowWhileLocked, LayerCreationCallback);
    void releaseLayer(SampleBufferDisplayLayerIdentifier);

    Ref<WorkQueue> protectedQueue() const { return m_queue; }

    ThreadSafeWeakPtr<GPUConnectionToWebProcess> m_connectionToWebProcess;
    Ref<IPC::Connection> m_connection;
    SharedPreferencesForWebProcess m_sharedPreferencesForWebProcess;
    Ref<WorkQueue> m_queue;
    mutable Lock m_layersLock;
    HashMap<SampleBufferDisplayLayerIdentifier, Ref<RemoteSampleBufferDisplayLayer>> m_layers WTF_GUARDED_BY_LOCK(m_layersLock);
};

}

#endif // PLATFORM(COCOA) && ENABLE(GPU_PROCESS) && ENABLE(MEDIA_STREAM)
