/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 8, 2023.
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

#include <WebCore/ResourceRequest.h>
#include <libsoup/soup.h>
#include <wtf/RunLoop.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/glib/GRefPtr.h>

namespace WebKit {
class NetworkSocketChannel;
struct SessionSet;

class WebSocketTask : public CanMakeWeakPtr<WebSocketTask>, public CanMakeCheckedPtr<WebSocketTask> {
    WTF_MAKE_TZONE_ALLOCATED(WebSocketTask);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(WebSocketTask);
public:
    WebSocketTask(NetworkSocketChannel&, const WebCore::ResourceRequest&, SoupSession*, SoupMessage*, const String& protocol);
    ~WebSocketTask();

    void sendString(std::span<const uint8_t>, CompletionHandler<void()>&&);
    void sendData(std::span<const uint8_t>, CompletionHandler<void()>&&);
    void close(int32_t code, const String& reason);

    void cancel();
    void resume();

    SessionSet* sessionSet() { return nullptr; }

private:
    void didConnect(GRefPtr<SoupWebsocketConnection>&&);
    void didFail(String&&);
    void didClose(unsigned short code, const String& reason);
    void delayFailTimerFired();

    String acceptedExtensions() const;

    Ref<NetworkSocketChannel> protectedChannel() const;

    static void didReceiveMessageCallback(WebSocketTask*, SoupWebsocketDataType, GBytes*);
    static void didReceiveErrorCallback(WebSocketTask*, GError*);
    static void didCloseCallback(WebSocketTask*);

    WeakRef<NetworkSocketChannel> m_channel;
    WebCore::ResourceRequest m_request;
    GRefPtr<SoupMessage> m_handshakeMessage;
    GRefPtr<SoupWebsocketConnection> m_connection;
    GRefPtr<GCancellable> m_cancellable;
    bool m_receivedDidFail { false };
    bool m_receivedDidClose { false };
    String m_delayErrorMessage;
    RunLoop::Timer m_delayFailTimer;
};

} // namespace WebKit
