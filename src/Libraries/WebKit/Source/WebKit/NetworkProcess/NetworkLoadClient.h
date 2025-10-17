/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 23, 2023.
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

#include <wtf/Forward.h>

namespace WebCore {
class AuthenticationChallenge;
class NetworkLoadMetrics;
class FragmentedSharedBuffer;
class ResourceError;
class ResourceRequest;
class ResourceResponse;
enum class PolicyAction : uint8_t;
}

namespace WebKit {

enum class PrivateRelayed : bool;
using ResponseCompletionHandler = CompletionHandler<void(WebCore::PolicyAction)>;

class NetworkLoadClient {
public:
    virtual ~NetworkLoadClient() { }

    virtual bool isSynchronous() const = 0;

    virtual bool isAllowedToAskUserForCredentials() const = 0;

    virtual void didSendData(uint64_t bytesSent, uint64_t totalBytesToBeSent) = 0;
    virtual void willSendRedirectedRequest(WebCore::ResourceRequest&&, WebCore::ResourceRequest&& redirectRequest, WebCore::ResourceResponse&& redirectResponse, CompletionHandler<void(WebCore::ResourceRequest&&)>&&) = 0;
    virtual void didReceiveInformationalResponse(WebCore::ResourceResponse&&) { };
    virtual void didReceiveResponse(WebCore::ResourceResponse&&, PrivateRelayed, ResponseCompletionHandler&&) = 0;
    virtual void didReceiveBuffer(const WebCore::FragmentedSharedBuffer&, uint64_t reportedEncodedDataLength) = 0;
    virtual void didFinishLoading(const WebCore::NetworkLoadMetrics&) = 0;
    virtual void didFailLoading(const WebCore::ResourceError&) = 0;
    virtual void didBlockAuthenticationChallenge() { };
    virtual void didReceiveChallenge(const WebCore::AuthenticationChallenge&) { };
    virtual bool shouldCaptureExtraNetworkLoadMetrics() const { return false; }
};

} // namespace WebKit
