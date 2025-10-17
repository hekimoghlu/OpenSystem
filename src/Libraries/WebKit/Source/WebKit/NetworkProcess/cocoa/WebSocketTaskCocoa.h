/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 6, 2024.
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

#include "NetworkTaskCocoa.h"
#include "WebPageProxyIdentifier.h"
#include <WebCore/FrameIdentifier.h>
#include <WebCore/PageIdentifier.h>
#include <WebCore/SecurityOriginData.h>
#include <wtf/RetainPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

OBJC_CLASS NSURLSessionWebSocketTask;

namespace WebKit {
class WebSocketTask;
}

namespace WebCore {
class ResourceResponse;
class ResourceRequest;
struct ClientOrigin;
}

namespace WebKit {
class NetworkSession;
class NetworkSessionCocoa;
class NetworkSocketChannel;
struct SessionSet;

class WebSocketTask : public CanMakeWeakPtr<WebSocketTask>, public CanMakeCheckedPtr<WebSocketTask>, public NetworkTaskCocoa {
    WTF_MAKE_TZONE_ALLOCATED(WebSocketTask);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(WebSocketTask);
public:
    WebSocketTask(NetworkSocketChannel&, WebPageProxyIdentifier, std::optional<WebCore::FrameIdentifier>, std::optional<WebCore::PageIdentifier>, WeakPtr<SessionSet>&&, const WebCore::ResourceRequest&, const WebCore::ClientOrigin&, RetainPtr<NSURLSessionWebSocketTask>&&, WebCore::StoredCredentialsPolicy);
    ~WebSocketTask();

    void sendString(std::span<const uint8_t>, CompletionHandler<void()>&&);
    void sendData(std::span<const uint8_t>, CompletionHandler<void()>&&);
    void close(int32_t code, const String& reason);

    void didConnect(const String&);
    void didClose(unsigned short code, const String& reason);

    void cancel();
    void resume();

    typedef uint64_t TaskIdentifier;
    TaskIdentifier identifier() const;

    NetworkSessionCocoa* networkSession();
    SessionSet* sessionSet() { return m_sessionSet.get(); }

    std::optional<WebCore::FrameIdentifier> frameID() const final { return m_frameID; }
    std::optional<WebCore::PageIdentifier> pageID() const final { return m_pageID; }
    std::optional<WebPageProxyIdentifier> webPageProxyID() const final { return m_webProxyPageID; }
    String partition() const { return m_partition; }
    const WebCore::SecurityOriginData& topOrigin() const { return m_topOrigin; }

private:
    void readNextMessage();

    RefPtr<NetworkSocketChannel> protectedChannel() const;

    NSURLSessionTask* task() const final;
    WebCore::StoredCredentialsPolicy storedCredentialsPolicy() const final { return m_storedCredentialsPolicy; }

    WeakPtr<NetworkSocketChannel> m_channel;
    RetainPtr<NSURLSessionWebSocketTask> m_task;
    bool m_receivedDidClose { false };
    bool m_receivedDidConnect { false };
    Markable<WebPageProxyIdentifier> m_webProxyPageID;
    std::optional<WebCore::FrameIdentifier> m_frameID;
    std::optional<WebCore::PageIdentifier> m_pageID;
    WeakPtr<SessionSet> m_sessionSet;
    String m_partition;
    WebCore::StoredCredentialsPolicy m_storedCredentialsPolicy { WebCore::StoredCredentialsPolicy::DoNotUse };
    WebCore::SecurityOriginData m_topOrigin;
};

} // namespace WebKit
