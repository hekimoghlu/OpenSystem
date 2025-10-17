/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 8, 2023.
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
#include "config.h"
#include "RemoteSampleBufferDisplayLayerManager.h"

#if PLATFORM(COCOA) && ENABLE(GPU_PROCESS) && ENABLE(MEDIA_STREAM)

#include "Decoder.h"
#include "GPUConnectionToWebProcess.h"
#include "GPUProcess.h"
#include "RemoteSampleBufferDisplayLayer.h"
#include "RemoteSampleBufferDisplayLayerManagerMessages.h"
#include "RemoteSampleBufferDisplayLayerMessages.h"
#include <WebCore/IntSize.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteSampleBufferDisplayLayerManager);

RemoteSampleBufferDisplayLayerManager::RemoteSampleBufferDisplayLayerManager(GPUConnectionToWebProcess& gpuConnectionToWebProcess, SharedPreferencesForWebProcess& sharedPreferencesForWebProcess)
    : m_connectionToWebProcess(gpuConnectionToWebProcess)
    , m_connection(gpuConnectionToWebProcess.connection())
    , m_sharedPreferencesForWebProcess(sharedPreferencesForWebProcess)
    , m_queue(gpuConnectionToWebProcess.protectedGPUProcess()->videoMediaStreamTrackRendererQueue())
{
    protectedQueue()->dispatch([this, protectedThis = Ref { *this }, sharedPreferencesForWebProcess] {
        m_sharedPreferencesForWebProcess = sharedPreferencesForWebProcess;
    });
}

void RemoteSampleBufferDisplayLayerManager::startListeningForIPC()
{
    auto connection = m_connectionToWebProcess.get();
    if (!connection)
        return;
    Ref ipcConnection = connection->protectedConnection();
    ipcConnection->addWorkQueueMessageReceiver(Messages::RemoteSampleBufferDisplayLayer::messageReceiverName(), m_queue, *this);
    ipcConnection->addWorkQueueMessageReceiver(Messages::RemoteSampleBufferDisplayLayerManager::messageReceiverName(), m_queue, *this);
}

RemoteSampleBufferDisplayLayerManager::~RemoteSampleBufferDisplayLayerManager() = default;

void RemoteSampleBufferDisplayLayerManager::close()
{
    auto connection = m_connectionToWebProcess.get();
    if (!connection)
        return;
    Ref ipcConnection = connection->protectedConnection();
    ipcConnection->removeWorkQueueMessageReceiver(Messages::RemoteSampleBufferDisplayLayer::messageReceiverName());
    ipcConnection->removeWorkQueueMessageReceiver(Messages::RemoteSampleBufferDisplayLayerManager::messageReceiverName());
    protectedQueue()->dispatch([this, protectedThis = Ref { *this }] {
        Locker lock(m_layersLock);
        callOnMainRunLoop([layers = WTFMove(m_layers)] { });
    });
}

bool RemoteSampleBufferDisplayLayerManager::dispatchMessage(IPC::Connection& connection, IPC::Decoder& decoder)
{
    if (!ObjectIdentifier<SampleBufferDisplayLayerIdentifierType>::isValidIdentifier(decoder.destinationID()))
        return false;

    auto identifier = ObjectIdentifier<SampleBufferDisplayLayerIdentifierType>(decoder.destinationID());
    Locker lock(m_layersLock);
    if (RefPtr layer = m_layers.get(identifier))
        layer->didReceiveMessage(connection, decoder);
    return true;
}

void RemoteSampleBufferDisplayLayerManager::createLayer(SampleBufferDisplayLayerIdentifier identifier, bool hideRootLayer, WebCore::IntSize size, bool shouldMaintainAspectRatio, bool canShowWhileLocked, LayerCreationCallback&& callback)
{
    callOnMainRunLoop([this, protectedThis = Ref { *this }, identifier, hideRootLayer, size, shouldMaintainAspectRatio, canShowWhileLocked, callback = WTFMove(callback)]() mutable {
        auto connection = m_connectionToWebProcess.get();
        if (!connection)
            return callback({ });
        auto layer = RemoteSampleBufferDisplayLayer::create(*connection, identifier, m_connection.copyRef(), protectedThis);
        if (!layer) {
            callback({ });
            return;
        }
        layer->initialize(hideRootLayer, size, shouldMaintainAspectRatio, canShowWhileLocked, [this, protectedThis = Ref { *this }, callback = WTFMove(callback), identifier, layer = Ref { *layer }](auto layerId) mutable {
            protectedQueue()->dispatch([this, protectedThis = WTFMove(protectedThis), callback = WTFMove(callback), identifier, layer = WTFMove(layer), layerId = WTFMove(layerId)]() mutable {
                Locker lock(m_layersLock);
                ASSERT(!m_layers.contains(identifier));
                m_layers.add(identifier, WTFMove(layer));
                callback(WTFMove(layerId));
            });
        });
    });
}

void RemoteSampleBufferDisplayLayerManager::releaseLayer(SampleBufferDisplayLayerIdentifier identifier)
{
    callOnMainRunLoop([this, protectedThis = Ref { *this }, identifier]() mutable {
        protectedQueue()->dispatch([this, protectedThis = WTFMove(protectedThis), identifier] {
            Locker lock(m_layersLock);
            ASSERT(m_layers.contains(identifier));
            callOnMainRunLoop([layer = m_layers.take(identifier)] { });
        });
    });
}

bool RemoteSampleBufferDisplayLayerManager::allowsExitUnderMemoryPressure() const
{
    Locker lock(m_layersLock);
    return m_layers.isEmpty();
}

void RemoteSampleBufferDisplayLayerManager::updateSampleBufferDisplayLayerBoundsAndPosition(SampleBufferDisplayLayerIdentifier identifier, WebCore::FloatRect bounds, std::optional<MachSendRight>&& sendRight)
{
    Locker lock(m_layersLock);
    if (RefPtr layer = m_layers.get(identifier))
        layer->updateBoundsAndPosition(bounds, WTFMove(sendRight));
}

void RemoteSampleBufferDisplayLayerManager::updateSharedPreferencesForWebProcess(SharedPreferencesForWebProcess sharedPreferencesForWebProcess)
{
    protectedQueue()->dispatch([this, protectedThis = Ref { *this }, sharedPreferencesForWebProcess = WTFMove(sharedPreferencesForWebProcess)] {
        m_sharedPreferencesForWebProcess = sharedPreferencesForWebProcess;
    });
}

}

#endif // PLATFORM(COCOA) && ENABLE(GPU_PROCESS) && ENABLE(MEDIA_STREAM)
