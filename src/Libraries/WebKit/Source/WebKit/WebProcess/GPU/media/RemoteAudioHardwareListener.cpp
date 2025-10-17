/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 9, 2024.
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
#include "RemoteAudioHardwareListener.h"

#if ENABLE(GPU_PROCESS)

#include "GPUConnectionToWebProcessMessages.h"
#include "GPUProcessProxy.h"
#include "RemoteAudioHardwareListenerMessages.h"
#include "WebProcess.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteAudioHardwareListener);

Ref<RemoteAudioHardwareListener> RemoteAudioHardwareListener::create(AudioHardwareListener::Client& client)
{
    return adoptRef(*new RemoteAudioHardwareListener(client));
}

RemoteAudioHardwareListener::RemoteAudioHardwareListener(AudioHardwareListener::Client& client)
    : AudioHardwareListener(client)
    , m_gpuProcessConnection(WebProcess::singleton().ensureGPUProcessConnection())
{
    auto gpuProcessConnection = m_gpuProcessConnection.get();
    gpuProcessConnection->addClient(*this);
    gpuProcessConnection->messageReceiverMap().addMessageReceiver(Messages::RemoteAudioHardwareListener::messageReceiverName(), identifier().toUInt64(), *this);
    gpuProcessConnection->connection().send(Messages::GPUConnectionToWebProcess::CreateAudioHardwareListener(identifier()), { });
}

RemoteAudioHardwareListener::~RemoteAudioHardwareListener()
{
    if (auto gpuProcessConnection = m_gpuProcessConnection.get()) {
        gpuProcessConnection->messageReceiverMap().removeMessageReceiver(*this);
        gpuProcessConnection->connection().send(Messages::GPUConnectionToWebProcess::ReleaseAudioHardwareListener(identifier()), 0);
    }
}

void RemoteAudioHardwareListener::gpuProcessConnectionDidClose(GPUProcessConnection& connection)
{
    audioHardwareDidBecomeInactive();

    ASSERT_UNUSED(connection, &connection == m_gpuProcessConnection.get());
    if (auto gpuProcessConnection = m_gpuProcessConnection.get()) {
        gpuProcessConnection->messageReceiverMap().removeMessageReceiver(*this);
        m_gpuProcessConnection = nullptr;
    }
}

void RemoteAudioHardwareListener::audioHardwareDidBecomeActive()
{
    setHardwareActivity(AudioHardwareActivityType::IsActive);
    m_client.audioHardwareDidBecomeActive();
}

void RemoteAudioHardwareListener::audioHardwareDidBecomeInactive()
{
    setHardwareActivity(AudioHardwareActivityType::IsInactive);
    m_client.audioHardwareDidBecomeInactive();
}

void RemoteAudioHardwareListener::audioOutputDeviceChanged(size_t bufferSizeMinimum, size_t bufferSizeMaximum)
{
    setSupportedBufferSizes({ bufferSizeMinimum, bufferSizeMaximum });
    m_client.audioOutputDeviceChanged();
}

}

#endif
