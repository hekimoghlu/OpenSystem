/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 5, 2025.
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
#import "config.h"
#import "WebSocketTaskCocoa.h"

#import "NetworkSessionCocoa.h"
#import "NetworkSocketChannel.h"
#import <Foundation/NSURLSession.h>
#import <WebCore/ClientOrigin.h>
#import <WebCore/ResourceRequest.h>
#import <WebCore/ResourceResponse.h>
#import <WebCore/ThreadableWebSocketChannel.h>
#import <wtf/BlockPtr.h>
#import <wtf/TZoneMallocInlines.h>
#import <wtf/cocoa/SpanCocoa.h>

namespace WebKit {

using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebSocketTask);

WebSocketTask::WebSocketTask(NetworkSocketChannel& channel, WebPageProxyIdentifier webProxyPageID, std::optional<FrameIdentifier> frameID, std::optional<PageIdentifier> pageID, WeakPtr<SessionSet>&& sessionSet, const WebCore::ResourceRequest& request, const WebCore::ClientOrigin& clientOrigin, RetainPtr<NSURLSessionWebSocketTask>&& task, WebCore::StoredCredentialsPolicy storedCredentialsPolicy)
    : NetworkTaskCocoa(*channel.session())
    , m_channel(channel)
    , m_task(WTFMove(task))
    , m_webProxyPageID(webProxyPageID)
    , m_frameID(frameID)
    , m_pageID(pageID)
    , m_sessionSet(WTFMove(sessionSet))
    , m_partition(request.cachePartition())
    , m_storedCredentialsPolicy(storedCredentialsPolicy)
{
    // We use topOrigin in case of service worker websocket connections, for which pageID does not link to a real page.
    // In that case, let's only call the callback for same origin loads.
    if (clientOrigin.topOrigin == clientOrigin.clientOrigin)
        m_topOrigin = clientOrigin.topOrigin;

    bool shouldBlockCookies = storedCredentialsPolicy == WebCore::StoredCredentialsPolicy::EphemeralStateless;
    if (auto* networkStorageSession = networkSession() ? networkSession()->networkStorageSession() : nullptr) {
        if (!shouldBlockCookies)
            shouldBlockCookies = networkStorageSession->shouldBlockCookies(request, frameID, pageID, shouldRelaxThirdPartyCookieBlocking());
    }
    if (shouldBlockCookies)
        blockCookies();

    readNextMessage();
    protectedChannel()->didSendHandshakeRequest(ResourceRequest { [m_task currentRequest] });

#if HAVE(ALLOW_ONLY_PARTITIONED_COOKIES)
    updateTaskWithStoragePartitionIdentifier(request);
#endif
}

WebSocketTask::~WebSocketTask() = default;

RefPtr<NetworkSocketChannel> WebSocketTask::protectedChannel() const
{
    return m_channel.get();
}

void WebSocketTask::readNextMessage()
{
    [m_task receiveMessageWithCompletionHandler:makeBlockPtr([this, weakThis = WeakPtr { *this }](NSURLSessionWebSocketMessage* _Nullable message, NSError * _Nullable error) {
        if (!weakThis)
            return;

        RefPtr channel = m_channel.get();
        if (error) {
            // If closeCode is not zero, we are closing the connection and didClose will be called for us.
            if ([m_task closeCode])
                return;

            if (!m_receivedDidConnect) {
                ResourceResponse response { [m_task response] };
                if (!response.isNull())
                    channel->didReceiveHandshakeResponse(WTFMove(response));
            }

            channel->didReceiveMessageError([error localizedDescription]);
            didClose(WebCore::ThreadableWebSocketChannel::CloseEventCodeAbnormalClosure, emptyString());
            return;
        }
        if (message.type == NSURLSessionWebSocketMessageTypeString)
            channel->didReceiveText(message.string);
        else
            channel->didReceiveBinaryData(span(message.data));

        readNextMessage();
    }).get()];
}

void WebSocketTask::cancel()
{
    [m_task cancel];
}

void WebSocketTask::resume()
{
    [m_task resume];
}

void WebSocketTask::didConnect(const String& protocol)
{
    String extensionsValue;
    auto response = [m_task response];
    if (auto *httpResponse  = dynamic_objc_cast<NSHTTPURLResponse>(response))
        extensionsValue = [httpResponse  valueForHTTPHeaderField:@"Sec-WebSocket-Extensions"];

    m_receivedDidConnect = true;
    RefPtr channel = m_channel.get();
    channel->didConnect(protocol, extensionsValue);
    channel->didReceiveHandshakeResponse(ResourceResponse { [m_task response] });
}

void WebSocketTask::didClose(unsigned short code, const String& reason)
{
    if (m_receivedDidClose)
        return;

    m_receivedDidClose = true;
    protectedChannel()->didClose(code, reason);
}

void WebSocketTask::sendString(std::span<const uint8_t> utf8String, CompletionHandler<void()>&& callback)
{
    auto text = adoptNS([[NSString alloc] initWithBytes:utf8String.data() length:utf8String.size() encoding:NSUTF8StringEncoding]);
    if (!text) {
        callback();
        return;
    }
    auto message = adoptNS([[NSURLSessionWebSocketMessage alloc] initWithString:text.get()]);
    [m_task sendMessage:message.get() completionHandler:makeBlockPtr([callback = WTFMove(callback)](NSError * _Nullable) mutable {
        callback();
    }).get()];
}

void WebSocketTask::sendData(std::span<const uint8_t> data, CompletionHandler<void()>&& callback)
{
    RetainPtr nsData = toNSData(data);
    auto message = adoptNS([[NSURLSessionWebSocketMessage alloc] initWithData:nsData.get()]);
    [m_task sendMessage:message.get() completionHandler:makeBlockPtr([callback = WTFMove(callback)](NSError * _Nullable) mutable {
        callback();
    }).get()];
}

void WebSocketTask::close(int32_t code, const String& reason)
{
    if (code == WebCore::ThreadableWebSocketChannel::CloseEventCodeNotSpecified)
        code = NSURLSessionWebSocketCloseCodeInvalid;
    auto utf8 = reason.utf8();
    RetainPtr nsData = toNSData(utf8.span());
    if ([m_task respondsToSelector:@selector(_sendCloseCode:reason:)]) {
        [m_task _sendCloseCode:(NSURLSessionWebSocketCloseCode)code reason:nsData.get()];
        return;
    }
    [m_task cancelWithCloseCode:(NSURLSessionWebSocketCloseCode)code reason:nsData.get()];
}

WebSocketTask::TaskIdentifier WebSocketTask::identifier() const
{
    return [m_task taskIdentifier];
}

NetworkSessionCocoa* WebSocketTask::networkSession()
{
    return static_cast<NetworkSessionCocoa*>(protectedChannel()->session());
}

NSURLSessionTask* WebSocketTask::task() const
{
    return m_task.get();
}

}
