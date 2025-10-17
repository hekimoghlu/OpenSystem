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

#include "EventTarget.h"
#include "WebSocketFrame.h"
#include <wtf/Forward.h>
#include <wtf/ObjectIdentifier.h>

namespace WebCore {

class Document;
class WeakPtrImplWithEventTargetData;
class ResourceRequest;
class ResourceResponse;
class WebSocketChannel;
class WebSocketChannelInspector;

using WebSocketChannelIdentifier = AtomicObjectIdentifier<WebSocketChannel>;

class WEBCORE_EXPORT WebSocketChannelInspector {
public:
    explicit WebSocketChannelInspector(Document&);
    ~WebSocketChannelInspector();

    void didCreateWebSocket(const URL&) const;
    void willSendWebSocketHandshakeRequest(const ResourceRequest&) const;
    void didReceiveWebSocketHandshakeResponse(const ResourceResponse&) const;
    void didCloseWebSocket() const;
    void didReceiveWebSocketFrame(const WebSocketFrame&) const;
    void didSendWebSocketFrame(const WebSocketFrame&) const;
    void didReceiveWebSocketFrameError(const String& errorMessage) const;
    
    WebSocketChannelIdentifier progressIdentifier() const { return m_progressIdentifier; }

    static WebSocketFrame createFrame(std::span<const uint8_t> data, WebSocketFrame::OpCode);

private:
    WeakPtr<Document, WeakPtrImplWithEventTargetData> m_document;
    WebSocketChannelIdentifier m_progressIdentifier;
};

} // namespace WebCore
