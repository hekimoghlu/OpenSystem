/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 1, 2024.
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

#include "AuthenticationChallengeDisposition.h"
#include "NetworkSession.h"

namespace WebKit {

struct NetworkSessionCreationParameters;
class WebSocketTask;

class NetworkSessionCurl final : public NetworkSession {
public:
    static std::unique_ptr<NetworkSession> create(NetworkProcess& networkProcess, const NetworkSessionCreationParameters& parameters)
    {
        return makeUnique<NetworkSessionCurl>(networkProcess, parameters);
    }
    NetworkSessionCurl(NetworkProcess&, const NetworkSessionCreationParameters&);
    ~NetworkSessionCurl();

    void clearAlternativeServices(WallTime) override;

    void didReceiveChallenge(WebSocketTask&, WebCore::AuthenticationChallenge&&, CompletionHandler<void(WebKit::AuthenticationChallengeDisposition, const WebCore::Credential&)>&&);

private:
    std::unique_ptr<WebSocketTask> createWebSocketTask(WebPageProxyIdentifier, std::optional<WebCore::FrameIdentifier>, std::optional<WebCore::PageIdentifier>, NetworkSocketChannel&, const WebCore::ResourceRequest&, const String& protocol, const WebCore::ClientOrigin&, bool, bool, OptionSet<WebCore::AdvancedPrivacyProtections>, WebCore::StoredCredentialsPolicy) final;
};

} // namespace WebKit
