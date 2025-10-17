/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 29, 2025.
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

#if ENABLE(SERVER_PRECONNECT)

#include "NetworkLoadClient.h"
#include <WebCore/Timer.h>
#include <wtf/CompletionHandler.h>
#include <wtf/WeakPtr.h>

namespace WebKit {

class NetworkLoad;
class NetworkProcess;
class NetworkSession;

struct NetworkLoadParameters;

class PreconnectTask final : public NetworkLoadClient {
public:
    PreconnectTask(NetworkSession&, NetworkLoadParameters&&, CompletionHandler<void(const WebCore::ResourceError&, const WebCore::NetworkLoadMetrics&)>&&);
    ~PreconnectTask();

    void setH2PingCallback(const URL&, CompletionHandler<void(Expected<WTF::Seconds, WebCore::ResourceError>&&)>&&);
    void setTimeout(Seconds);
    void start();

private:
    // NetworkLoadClient.
    bool isSynchronous() const final { return false; }
    bool isAllowedToAskUserForCredentials() const final { return false; }
    void didSendData(uint64_t bytesSent, uint64_t totalBytesToBeSent) final;
    void willSendRedirectedRequest(WebCore::ResourceRequest&&, WebCore::ResourceRequest&& redirectRequest, WebCore::ResourceResponse&& redirectResponse, CompletionHandler<void(WebCore::ResourceRequest&&)>&&) final;
    void didReceiveResponse(WebCore::ResourceResponse&&, PrivateRelayed, ResponseCompletionHandler&&) final;
    void didReceiveBuffer(const WebCore::FragmentedSharedBuffer&, uint64_t reportedEncodedDataLength) final;
    void didFinishLoading(const WebCore::NetworkLoadMetrics&) final;
    void didFailLoading(const WebCore::ResourceError&) final;

    void didFinish(const WebCore::ResourceError&, const WebCore::NetworkLoadMetrics&);

    Ref<NetworkLoad> m_networkLoad;
    CompletionHandler<void(const WebCore::ResourceError&, const WebCore::NetworkLoadMetrics&)> m_completionHandler;
    Seconds m_timeout;
    WebCore::Timer m_timeoutTimer;
};

} // namespace WebKit

#endif // ENABLE(SERVER_PRECONNECT)
