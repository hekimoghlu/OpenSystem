/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 24, 2024.
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
#include "WebSocketDeflateFramer.h"

#include "WebSocketExtensionProcessor.h"
#include "WebSocketFrame.h"
#include <wtf/HashMap.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/StringHash.h>
#include <wtf/text/StringToIntegerConversion.h>

namespace WebCore {

class WebSocketExtensionDeflateFrame final : public WebSocketExtensionProcessor {
    WTF_MAKE_TZONE_ALLOCATED(WebSocketExtensionDeflateFrame);
public:
    explicit WebSocketExtensionDeflateFrame(WebSocketDeflateFramer&);

private:
    String handshakeString() final;
    bool processResponse(const HashMap<String, String>&) final;
    String failureReason() final { return m_failureReason; }

    WebSocketDeflateFramer& m_framer;
    bool m_responseProcessed { false };
    String m_failureReason;
};

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebSocketExtensionDeflateFrame);

// FXIME: Remove vendor prefix after the specification matured.
WebSocketExtensionDeflateFrame::WebSocketExtensionDeflateFrame(WebSocketDeflateFramer& framer)
    : WebSocketExtensionProcessor("x-webkit-deflate-frame"_s)
    , m_framer(framer)
{
}

String WebSocketExtensionDeflateFrame::handshakeString()
{
    return extensionToken(); // No parameter
}

bool WebSocketExtensionDeflateFrame::processResponse(const HashMap<String, String>& serverParameters)
{
    if (m_responseProcessed) {
        m_failureReason = "Received duplicate deflate-frame response"_s;
        return false;
    }
    m_responseProcessed = true;

    unsigned expectedNumParameters = 0;
    int windowBits = 15;
    auto parameter = serverParameters.find<HashTranslatorASCIILiteral>("max_window_bits"_s);
    if (parameter != serverParameters.end()) {
        windowBits = parseIntegerAllowingTrailingJunk<int>(parameter->value).value_or(0);
        if (windowBits < 8 || windowBits > 15) {
            m_failureReason = "Received invalid max_window_bits parameter"_s;
            return false;
        }
        expectedNumParameters++;
    }

    WebSocketDeflater::ContextTakeOverMode mode = WebSocketDeflater::TakeOverContext;
    parameter = serverParameters.find<HashTranslatorASCIILiteral>("no_context_takeover"_s);
    if (parameter != serverParameters.end()) {
        if (!parameter->value.isNull()) {
            m_failureReason = "Received invalid no_context_takeover parameter"_s;
            return false;
        }
        mode = WebSocketDeflater::DoNotTakeOverContext;
        expectedNumParameters++;
    }

    if (expectedNumParameters != serverParameters.size()) {
        m_failureReason = "Received unexpected deflate-frame parameter"_s;
        return false;
    }

    m_framer.enableDeflate(windowBits, mode);
    return true;
}

WTF_MAKE_TZONE_ALLOCATED_IMPL(DeflateResultHolder);

DeflateResultHolder::DeflateResultHolder(WebSocketDeflateFramer& framer)
    : m_framer(framer)
{
}

DeflateResultHolder::~DeflateResultHolder()
{
    m_framer.resetDeflateContext();
}

void DeflateResultHolder::fail(const String& failureReason)
{
    m_succeeded = false;
    m_failureReason = failureReason;
}

WTF_MAKE_TZONE_ALLOCATED_IMPL(InflateResultHolder);

InflateResultHolder::InflateResultHolder(WebSocketDeflateFramer& framer)
    : m_framer(framer)
{
}

InflateResultHolder::~InflateResultHolder()
{
    m_framer.resetInflateContext();
}

void InflateResultHolder::fail(const String& failureReason)
{
    m_succeeded = false;
    m_failureReason = failureReason;
}

std::unique_ptr<WebSocketExtensionProcessor> WebSocketDeflateFramer::createExtensionProcessor()
{
    return makeUnique<WebSocketExtensionDeflateFrame>(*this);
}

void WebSocketDeflateFramer::enableDeflate(int windowBits, WebSocketDeflater::ContextTakeOverMode mode)
{
    m_deflater = makeUnique<WebSocketDeflater>(windowBits, mode);
    m_inflater = makeUnique<WebSocketInflater>();
    if (!m_deflater->initialize() || !m_inflater->initialize()) {
        m_deflater = nullptr;
        m_inflater = nullptr;
        return;
    }
    m_enabled = true;
}

std::unique_ptr<DeflateResultHolder> WebSocketDeflateFramer::deflate(WebSocketFrame& frame)
{
    auto result = makeUnique<DeflateResultHolder>(*this);
    if (!enabled() || !WebSocketFrame::isNonControlOpCode(frame.opCode) || !frame.payload.size())
        return result;
    if (!m_deflater->addBytes(frame.payload) || !m_deflater->finish()) {
        result->fail("Failed to compress frame"_s);
        return result;
    }
    frame.compress = true;
    frame.payload = m_deflater->span();
    return result;
}

void WebSocketDeflateFramer::resetDeflateContext()
{
    if (m_deflater)
        m_deflater->reset();
}

std::unique_ptr<InflateResultHolder> WebSocketDeflateFramer::inflate(WebSocketFrame& frame)
{
    auto result = makeUnique<InflateResultHolder>(*this);
    if (!enabled() && frame.compress) {
        result->fail("Compressed bit must be 0 if no negotiated deflate-frame extension"_s);
        return result;
    }
    if (!frame.compress)
        return result;
    if (!WebSocketFrame::isNonControlOpCode(frame.opCode)) {
        result->fail("Received unexpected compressed frame"_s);
        return result;
    }
    if (!m_inflater->addBytes(frame.payload) || !m_inflater->finish()) {
        result->fail("Failed to decompress frame"_s);
        return result;
    }
    frame.compress = false;
    frame.payload = m_inflater->span();
    return result;
}

void WebSocketDeflateFramer::resetInflateContext()
{
    if (m_inflater)
        m_inflater->reset();
}

void WebSocketDeflateFramer::didFail()
{
    resetDeflateContext();
    resetInflateContext();
}

} // namespace WebCore
