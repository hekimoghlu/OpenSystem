/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 10, 2025.
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
#include "AudioSessionRoutingArbitrator.h"

#if ENABLE(ROUTING_ARBITRATION)

#include "AudioSessionRoutingArbitratorProxy.h"
#include "AudioSessionRoutingArbitratorProxyMessages.h"
#include "WebProcess.h"
#include <wtf/LoggerHelper.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(AudioSessionRoutingArbitrator);

AudioSessionRoutingArbitrator::AudioSessionRoutingArbitrator(WebProcess& process)
    : m_observer([this] (AudioSession& session) { session.setRoutingArbitrationClient(*this); })
    , m_logIdentifier(LoggerHelper::uniqueLogIdentifier())
{
    AudioSession::addAudioSessionChangedObserver(m_observer);
}

AudioSessionRoutingArbitrator::~AudioSessionRoutingArbitrator() = default;

void AudioSessionRoutingArbitrator::beginRoutingArbitrationWithCategory(AudioSession::CategoryType category, CompletionHandler<void(RoutingArbitrationError, DefaultRouteChanged)>&& callback)
{
    WebProcess::singleton().parentProcessConnection()->sendWithAsyncReply(Messages::AudioSessionRoutingArbitratorProxy::BeginRoutingArbitrationWithCategory(category), WTFMove(callback), AudioSessionRoutingArbitratorProxy::destinationId());
}

void AudioSessionRoutingArbitrator::leaveRoutingAbritration()
{
    WebProcess::singleton().parentProcessConnection()->send(Messages::AudioSessionRoutingArbitratorProxy::EndRoutingArbitration(), AudioSessionRoutingArbitratorProxy::destinationId());
}

bool AudioSessionRoutingArbitrator::canLog() const
{
    return WebProcess::singleton().sessionID().isAlwaysOnLoggingAllowed();
}

}

#endif
