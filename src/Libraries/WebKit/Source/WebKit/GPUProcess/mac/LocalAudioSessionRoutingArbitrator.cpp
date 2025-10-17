/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 12, 2023.
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
#include "LocalAudioSessionRoutingArbitrator.h"

#include "GPUConnectionToWebProcess.h"
#include "GPUProcess.h"
#include "GPUProcessConnectionMessages.h"
#include "Logging.h"
#include <wtf/LoggerHelper.h>
#include <wtf/TZoneMallocInlines.h>

#if ENABLE(ROUTING_ARBITRATION) && HAVE(AVAUDIO_ROUTING_ARBITER)

namespace WebKit {

using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(LocalAudioSessionRoutingArbitrator);

std::unique_ptr<LocalAudioSessionRoutingArbitrator> LocalAudioSessionRoutingArbitrator::create(GPUConnectionToWebProcess& gpuConnectionToWebProcess)
{
    return makeUnique<LocalAudioSessionRoutingArbitrator>(gpuConnectionToWebProcess);
}

LocalAudioSessionRoutingArbitrator::LocalAudioSessionRoutingArbitrator(GPUConnectionToWebProcess& gpuConnectionToWebProcess)
    : m_connectionToWebProcess(gpuConnectionToWebProcess)
    , m_logIdentifier(LoggerHelper::uniqueLogIdentifier())
{
}

LocalAudioSessionRoutingArbitrator::~LocalAudioSessionRoutingArbitrator() = default;

void LocalAudioSessionRoutingArbitrator::processDidTerminate()
{
    leaveRoutingAbritration();
}

void LocalAudioSessionRoutingArbitrator::beginRoutingArbitrationWithCategory(AudioSession::CategoryType category, CompletionHandler<void(RoutingArbitrationError, DefaultRouteChanged)>&& callback)
{
    ALWAYS_LOG(LOGIDENTIFIER, category);
    RefPtr connection = m_connectionToWebProcess.get();
    if (!connection)
        return;
    connection->protectedConnection()->sendWithAsyncReply(Messages::GPUProcessConnection::BeginRoutingArbitrationWithCategory(category), WTFMove(callback), 0);
}

void LocalAudioSessionRoutingArbitrator::leaveRoutingAbritration()
{
    RefPtr connection = m_connectionToWebProcess.get();
    if (!connection)
        return;
    connection->protectedConnection()->send(Messages::GPUProcessConnection::EndRoutingArbitration(), 0);
}

Logger& LocalAudioSessionRoutingArbitrator::logger()
{
    return m_connectionToWebProcess.get()->logger();
};

WTFLogChannel& LocalAudioSessionRoutingArbitrator::logChannel() const
{
    return WebKit2LogMedia;
}

bool LocalAudioSessionRoutingArbitrator::canLog() const
{
    if (RefPtr connection = m_connectionToWebProcess.get())
        return connection->isAlwaysOnLoggingAllowed();
    return false;
}

}

#endif
