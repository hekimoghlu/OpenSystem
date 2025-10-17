/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 2, 2023.
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
#include "WebSocketChannelInspector.h"

#include "Document.h"
#include "InspectorInstrumentation.h"
#include "Page.h"
#include "ProgressTracker.h"
#include "WebSocketFrame.h"

namespace WebCore {

WebSocketChannelInspector::WebSocketChannelInspector(Document& document)
    : m_document(document)
    , m_progressIdentifier(WebSocketChannelIdentifier::generate())
{
}

WebSocketChannelInspector::~WebSocketChannelInspector() = default;

void WebSocketChannelInspector::didCreateWebSocket(const URL& url) const
{
    if (!m_document)
        return;

    InspectorInstrumentation::didCreateWebSocket(m_document.get(), m_progressIdentifier, url);
}

void WebSocketChannelInspector::willSendWebSocketHandshakeRequest(const ResourceRequest& request) const
{
    if (!m_document)
        return;

    InspectorInstrumentation::willSendWebSocketHandshakeRequest(m_document.get(), m_progressIdentifier, request);
}

void WebSocketChannelInspector::didReceiveWebSocketHandshakeResponse(const ResourceResponse& response) const
{
    if (!m_document)
        return;

    InspectorInstrumentation::didReceiveWebSocketHandshakeResponse(m_document.get(), m_progressIdentifier, response);
}

void WebSocketChannelInspector::didCloseWebSocket() const
{
    if (!m_document)
        return;

    InspectorInstrumentation::didCloseWebSocket(m_document.get(), m_progressIdentifier);
}

void WebSocketChannelInspector::didReceiveWebSocketFrame(const WebSocketFrame& frame) const
{
    if (!m_document)
        return;

    InspectorInstrumentation::didReceiveWebSocketFrame(m_document.get(), m_progressIdentifier, frame);
}

void WebSocketChannelInspector::didSendWebSocketFrame(const WebSocketFrame& frame) const
{
    if (!m_document)
        return;

    InspectorInstrumentation::didSendWebSocketFrame(m_document.get(), m_progressIdentifier, frame);
}

void WebSocketChannelInspector::didReceiveWebSocketFrameError(const String& errorMessage) const
{
    if (!m_document)
        return;

    InspectorInstrumentation::didReceiveWebSocketFrameError(m_document.get(), m_progressIdentifier, errorMessage);
}

WebSocketFrame WebSocketChannelInspector::createFrame(std::span<const uint8_t> data, WebSocketFrame::OpCode opCode)
{
    // This is an approximation since frames can be merged on a single message.
    WebSocketFrame frame;
    frame.opCode = opCode;
    frame.masked = false;
    frame.payload = data;

    // WebInspector does not use them.
    frame.final = false;
    frame.compress = false;
    frame.reserved2 = false;
    frame.reserved3 = false;

    return frame;
}

} // namespace WebCore
