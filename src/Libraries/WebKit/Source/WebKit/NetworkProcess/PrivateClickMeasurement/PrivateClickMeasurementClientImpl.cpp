/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 27, 2021.
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
#include "PrivateClickMeasurementClientImpl.h"

#include "NetworkProcess.h"
#include "NetworkSession.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebKit::PCM {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ClientImpl);

ClientImpl::ClientImpl(NetworkSession& session, NetworkProcess& networkProcess)
    : m_networkSession(session)
    , m_networkProcess(networkProcess) { }

void ClientImpl::broadcastConsoleMessage(JSC::MessageLevel messageLevel, const String& message)
{
    if (!featureEnabled())
        return;

    m_networkProcess->broadcastConsoleMessage(m_networkSession->sessionID(), MessageSource::PrivateClickMeasurement, messageLevel, message);
}

bool ClientImpl::featureEnabled() const
{
    return m_networkSession && m_networkProcess->privateClickMeasurementEnabled();
}

bool ClientImpl::debugModeEnabled() const
{
    return m_networkSession && m_networkSession->privateClickMeasurementDebugModeEnabled();
}

bool ClientImpl::usesEphemeralDataStore() const
{
    return m_networkSession && m_networkSession->sessionID().isEphemeral();
}

} // namespace WebKit::PCM
