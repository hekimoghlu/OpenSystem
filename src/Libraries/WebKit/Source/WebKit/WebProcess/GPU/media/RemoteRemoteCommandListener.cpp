/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 10, 2023.
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
#include "RemoteRemoteCommandListener.h"

#if ENABLE(GPU_PROCESS)

#include "GPUConnectionToWebProcessMessages.h"
#include "GPUProcessProxy.h"
#include "RemoteRemoteCommandListenerMessages.h"
#include "RemoteRemoteCommandListenerProxyMessages.h"
#include "WebProcess.h"

namespace WebKit {

using namespace WebCore;

Ref<RemoteRemoteCommandListener> RemoteRemoteCommandListener::create(RemoteCommandListenerClient& client)
{
    return adoptRef(*new RemoteRemoteCommandListener(client));
}

RemoteRemoteCommandListener::RemoteRemoteCommandListener(RemoteCommandListenerClient& client)
    : RemoteCommandListener(client)
{
    ensureGPUProcessConnection();
}

RemoteRemoteCommandListener::~RemoteRemoteCommandListener()
{
    if (RefPtr gpuProcessConnection = m_gpuProcessConnection.get()) {
        gpuProcessConnection->messageReceiverMap().removeMessageReceiver(Messages::RemoteRemoteCommandListener::messageReceiverName(), identifier().toUInt64());
        gpuProcessConnection->connection().send(Messages::GPUConnectionToWebProcess::ReleaseRemoteCommandListener(identifier()), 0);
    }
}

GPUProcessConnection& RemoteRemoteCommandListener::ensureGPUProcessConnection()
{
    RefPtr gpuProcessConnection = m_gpuProcessConnection.get();
    if (!gpuProcessConnection) {
        gpuProcessConnection = &WebProcess::singleton().ensureGPUProcessConnection();
        m_gpuProcessConnection = gpuProcessConnection;
        gpuProcessConnection->addClient(*this);
        gpuProcessConnection->messageReceiverMap().addMessageReceiver(Messages::RemoteRemoteCommandListener::messageReceiverName(), identifier().toUInt64(), *this);
        gpuProcessConnection->connection().send(Messages::GPUConnectionToWebProcess::CreateRemoteCommandListener(identifier()), { });
    }
    return *gpuProcessConnection;
}

void RemoteRemoteCommandListener::gpuProcessConnectionDidClose(GPUProcessConnection& gpuProcessConnection)
{
    ASSERT(m_gpuProcessConnection.get() == &gpuProcessConnection);

    gpuProcessConnection.messageReceiverMap().removeMessageReceiver(Messages::RemoteRemoteCommandListener::messageReceiverName(), identifier().toUInt64());
    m_gpuProcessConnection = nullptr;

    // FIXME: GPUProcess will be relaunched/RemoteCommandListener re-created when calling updateSupportedCommands().
    // FIXME: Should we relaunch the GPUProcess pro-actively and re-create the RemoteCommandListener?
}

void RemoteRemoteCommandListener::didReceiveRemoteControlCommand(WebCore::PlatformMediaSession::RemoteControlCommandType type, const PlatformMediaSession::RemoteCommandArgument& argument)
{
    client().didReceiveRemoteControlCommand(type, argument);
}

void RemoteRemoteCommandListener::updateSupportedCommands()
{
    auto& supportedCommands = this->supportedCommands();
    if (m_currentCommands == supportedCommands && m_currentSupportSeeking == supportsSeeking())
        return;

    m_currentCommands = supportedCommands;
    m_currentSupportSeeking = supportsSeeking();

    ensureGPUProcessConnection().connection().send(Messages::RemoteRemoteCommandListenerProxy::UpdateSupportedCommands { copyToVector(supportedCommands), m_currentSupportSeeking }, identifier());
}

}

#endif
