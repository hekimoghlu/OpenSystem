/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 12, 2023.
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

class Document;
class ResourceRequest;
class ResourceResponse;
class WebSocketChannel;
struct WebSocketFrame;
using WebSocketChannelIdentifier = AtomicObjectIdentifier<WebSocketChannel>;

class WEBCORE_EXPORT LegacyWebSocketInspectorInstrumentation {
public:
    static bool hasFrontends();
    static void didCreateWebSocket(Document*, WebSocketChannelIdentifier, const URL& requestURL);
    static void willSendWebSocketHandshakeRequest(Document*, WebSocketChannelIdentifier, const ResourceRequest&);
    static void didReceiveWebSocketHandshakeResponse(Document*, WebSocketChannelIdentifier, const ResourceResponse&);
    static void didCloseWebSocket(Document*, WebSocketChannelIdentifier);
    static void didReceiveWebSocketFrame(Document*, WebSocketChannelIdentifier, const WebSocketFrame&);
    static void didSendWebSocketFrame(Document*, WebSocketChannelIdentifier, const WebSocketFrame&);
    static void didReceiveWebSocketFrameError(Document*, WebSocketChannelIdentifier, const String& errorMessage);
};

} // namespace WebCore
