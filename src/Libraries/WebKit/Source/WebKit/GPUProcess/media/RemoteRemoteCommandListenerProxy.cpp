/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 20, 2022.
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
#include "RemoteRemoteCommandListenerProxy.h"

#if ENABLE(GPU_PROCESS)

#include "GPUConnectionToWebProcess.h"
#include "GPUProcess.h"
#include "RemoteRemoteCommandListenerMessages.h"
#include "SharedPreferencesForWebProcess.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteRemoteCommandListenerProxy);

RemoteRemoteCommandListenerProxy::RemoteRemoteCommandListenerProxy(GPUConnectionToWebProcess& gpuConnection, RemoteRemoteCommandListenerIdentifier&& identifier)
    : m_gpuConnection(gpuConnection)
    , m_identifier(WTFMove(identifier))
{
}

RemoteRemoteCommandListenerProxy::~RemoteRemoteCommandListenerProxy() = default;

void RemoteRemoteCommandListenerProxy::updateSupportedCommands(Vector<WebCore::PlatformMediaSession::RemoteControlCommandType>&& registeredCommands, bool supportsSeeking)
{
    m_supportedCommands.clear();
    m_supportedCommands.add(registeredCommands.begin(), registeredCommands.end());
    m_supportsSeeking = supportsSeeking;

    if (auto connection = m_gpuConnection.get())
        connection->updateSupportedRemoteCommands();
}

std::optional<SharedPreferencesForWebProcess> RemoteRemoteCommandListenerProxy::sharedPreferencesForWebProcess() const
{
    if (RefPtr gpuConnectionToWebProcess = m_gpuConnection.get())
        return gpuConnectionToWebProcess->sharedPreferencesForWebProcess();

    return std::nullopt;
}

}

#endif
