/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 24, 2024.
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
#include "LegacyWebSocketInspectorInstrumentation.h"

#include "InspectorInstrumentation.h"

namespace WebCore {

bool LegacyWebSocketInspectorInstrumentation::hasFrontends()
{
    return InspectorInstrumentation::hasFrontends();
}

void LegacyWebSocketInspectorInstrumentation::didCreateWebSocket(Document* document, WebSocketChannelIdentifier identifier, const URL& requestURL)
{
    InspectorInstrumentation::didCreateWebSocket(document, identifier, requestURL);
}

void LegacyWebSocketInspectorInstrumentation::willSendWebSocketHandshakeRequest(Document* document, WebSocketChannelIdentifier identifier, const ResourceRequest& request)
{
    InspectorInstrumentation::willSendWebSocketHandshakeRequest(document, identifier, request);
}

void LegacyWebSocketInspectorInstrumentation::didReceiveWebSocketHandshakeResponse(Document* document, WebSocketChannelIdentifier identifier, const ResourceResponse& response)
{
    InspectorInstrumentation::didReceiveWebSocketHandshakeResponse(document, identifier, response);
}

void LegacyWebSocketInspectorInstrumentation::didCloseWebSocket(Document* document, WebSocketChannelIdentifier identifier)
{
    InspectorInstrumentation::didCloseWebSocket(document, identifier);
}

void LegacyWebSocketInspectorInstrumentation::didReceiveWebSocketFrame(Document* document, WebSocketChannelIdentifier identifier, const WebSocketFrame& frame)
{
    InspectorInstrumentation::didReceiveWebSocketFrame(document, identifier, frame);
}

void LegacyWebSocketInspectorInstrumentation::didSendWebSocketFrame(Document* document, WebSocketChannelIdentifier identifier, const WebSocketFrame& frame)
{
    InspectorInstrumentation::didSendWebSocketFrame(document, identifier, frame);
}

void LegacyWebSocketInspectorInstrumentation::didReceiveWebSocketFrameError(Document* document, WebSocketChannelIdentifier identifier, const String& errorMessage)
{
    InspectorInstrumentation::didReceiveWebSocketFrameError(document, identifier, errorMessage);
}

}
