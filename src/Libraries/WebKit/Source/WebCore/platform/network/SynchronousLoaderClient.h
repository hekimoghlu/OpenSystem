/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 16, 2022.
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

#include "ResourceError.h"
#include "ResourceHandleClient.h"
#include "ResourceResponse.h"
#include <wtf/Function.h>
#include <wtf/MessageQueue.h>
#include <wtf/ThreadSafeRefCounted.h>

namespace WebCore {

class SharedBuffer;

class SynchronousLoaderMessageQueue : public ThreadSafeRefCounted<SynchronousLoaderMessageQueue> {
public:
    static Ref<SynchronousLoaderMessageQueue> create() { return adoptRef(*new SynchronousLoaderMessageQueue); }

    void append(std::unique_ptr<Function<void()>>&& task) { m_queue.append(WTFMove(task)); }
    void kill() { m_queue.kill(); }
    bool killed() const { return m_queue.killed(); }
    std::unique_ptr<Function<void()>> waitForMessage() { return m_queue.waitForMessage(); }

private:
    SynchronousLoaderMessageQueue() = default;
    MessageQueue<Function<void()>> m_queue;
};

class SynchronousLoaderClient final : public ResourceHandleClient {
public:
    SynchronousLoaderClient();
    virtual ~SynchronousLoaderClient();

    void setAllowStoredCredentials(bool allow) { m_allowStoredCredentials = allow; }
    const ResourceResponse& response() const { return m_response; }
    Vector<uint8_t>& mutableData() { return m_data; }
    const ResourceError& error() const { return m_error; }
    SynchronousLoaderMessageQueue& messageQueue() { return m_messageQueue.get(); }

    WEBCORE_EXPORT static ResourceError platformBadResponseError();

private:
    void willSendRequestAsync(ResourceHandle*, ResourceRequest&&, ResourceResponse&&, CompletionHandler<void(ResourceRequest&&)>&&) override;
    bool shouldUseCredentialStorage(ResourceHandle*) override;
    void didReceiveAuthenticationChallenge(ResourceHandle*, const AuthenticationChallenge&) override;
    void didReceiveResponseAsync(ResourceHandle*, ResourceResponse&&, CompletionHandler<void()>&&) override;
    void didReceiveData(ResourceHandle*, const SharedBuffer&, int /*encodedDataLength*/) override;
    void didFinishLoading(ResourceHandle*, const NetworkLoadMetrics&) override;
    void didFail(ResourceHandle*, const ResourceError&) override;
#if USE(PROTECTION_SPACE_AUTH_CALLBACK)
    void canAuthenticateAgainstProtectionSpaceAsync(ResourceHandle*, const ProtectionSpace&, CompletionHandler<void(bool)>&&) override;
#endif

    bool m_allowStoredCredentials { false };
    ResourceResponse m_response;
    Vector<uint8_t> m_data;
    ResourceError m_error;
    Ref<SynchronousLoaderMessageQueue> m_messageQueue;
};
}
