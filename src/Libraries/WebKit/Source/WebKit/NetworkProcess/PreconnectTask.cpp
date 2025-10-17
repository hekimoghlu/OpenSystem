/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 4, 2024.
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
#include "PreconnectTask.h"

#if ENABLE(SERVER_PRECONNECT)

#include "Logging.h"
#include "NetworkLoad.h"
#include "NetworkLoadParameters.h"
#include "NetworkProcess.h"
#include "WebErrors.h"
#include <WebCore/ResourceError.h>

namespace WebKit {

using namespace WebCore;

PreconnectTask::PreconnectTask(NetworkSession& networkSession, NetworkLoadParameters&& parameters, CompletionHandler<void(const ResourceError&, const WebCore::NetworkLoadMetrics&)>&& completionHandler)
    : m_networkLoad(NetworkLoad::create(*this, WTFMove(parameters), networkSession))
    , m_completionHandler(WTFMove(completionHandler))
    , m_timeout(60_s)
    , m_timeoutTimer([this] { didFinish(ResourceError { String(), 0, m_networkLoad->parameters().request.url(), "Preconnection timed out"_s, ResourceError::Type::Timeout }, { }); })
{
    RELEASE_LOG(Network, "%p - PreconnectTask::PreconnectTask()", this);

    ASSERT(m_networkLoad->parameters().shouldPreconnectOnly == PreconnectOnly::Yes);
}

void PreconnectTask::setH2PingCallback(const URL& url, CompletionHandler<void(Expected<WTF::Seconds, WebCore::ResourceError>&&)>&& completionHandler)
{
    m_networkLoad->setH2PingCallback(url, WTFMove(completionHandler));
}

void PreconnectTask::setTimeout(Seconds timeout)
{
    m_timeout = timeout;
}

void PreconnectTask::start()
{
    m_timeoutTimer.startOneShot(m_timeout);
    m_networkLoad->start();
}

PreconnectTask::~PreconnectTask() = default;

void PreconnectTask::willSendRedirectedRequest(ResourceRequest&&, ResourceRequest&& redirectRequest, ResourceResponse&& redirectResponse, CompletionHandler<void(WebCore::ResourceRequest&&)>&& completionHandler)
{
    // HSTS "redirection" may happen here.
#if ASSERT_ENABLED
    auto url = redirectResponse.url();
    ASSERT(url.protocol() == "http"_s);
    url.setProtocol("https"_s);
    ASSERT(redirectRequest.url() == url);
#endif
    completionHandler(WTFMove(redirectRequest));
}

void PreconnectTask::didReceiveResponse(ResourceResponse&& response, PrivateRelayed, ResponseCompletionHandler&& completionHandler)
{
    ASSERT_NOT_REACHED();
    completionHandler(PolicyAction::Ignore);
}

void PreconnectTask::didReceiveBuffer(const FragmentedSharedBuffer&, uint64_t reportedEncodedDataLength)
{
    ASSERT_NOT_REACHED();
}

void PreconnectTask::didFinishLoading(const NetworkLoadMetrics& metrics)
{
    RELEASE_LOG(Network, "%p - PreconnectTask::didFinishLoading", this);
    didFinish({ }, metrics);
}

void PreconnectTask::didFailLoading(const ResourceError& error)
{
    RELEASE_LOG(Network, "%p - PreconnectTask::didFailLoading, error_code=%d", this, error.errorCode());
    didFinish(error, NetworkLoadMetrics::emptyMetrics());
}

void PreconnectTask::didSendData(uint64_t bytesSent, uint64_t totalBytesToBeSent)
{
    ASSERT_NOT_REACHED();
}

void PreconnectTask::didFinish(const ResourceError& error, const NetworkLoadMetrics& metrics)
{
    if (m_completionHandler)
        m_completionHandler(error, metrics);
    delete this;
}

} // namespace WebKit

#endif // ENABLE(SERVER_PRECONNECT)
