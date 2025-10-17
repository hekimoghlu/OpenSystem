/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 29, 2025.
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

#include "WebSocketExtensionDispatcher.h"
#include "WebSocketExtensionParser.h"
#include <wtf/ASCIICType.h>
#include <wtf/HashMap.h>
#include <wtf/text/CString.h>
#include <wtf/text/MakeString.h>
#include <wtf/text/StringHash.h>

namespace WebCore {

void WebSocketExtensionDispatcher::reset()
{
    m_processors.clear();
}

void WebSocketExtensionDispatcher::addProcessor(std::unique_ptr<WebSocketExtensionProcessor> processor)
{
    for (auto& extensionProcessor : m_processors) {
        if (extensionProcessor->extensionToken() == processor->extensionToken())
            return;
    }
    ASSERT(processor->handshakeString().length());
    ASSERT(!processor->handshakeString().contains('\n'));
    ASSERT(!processor->handshakeString().contains(static_cast<UChar>('\0')));
    m_processors.append(WTFMove(processor));
}

const String WebSocketExtensionDispatcher::createHeaderValue() const
{
    size_t numProcessors = m_processors.size();
    if (!numProcessors)
        return String();

    return makeString(interleave(m_processors, [](auto& processor) { return processor->handshakeString(); }, ", "_s));
}

void WebSocketExtensionDispatcher::appendAcceptedExtension(const String& extensionToken, HashMap<String, String>& extensionParameters)
{
    m_acceptedExtensionsBuilder.append(m_acceptedExtensionsBuilder.isEmpty() ? ""_s : ", "_s, extensionToken);
    // FIXME: Should use ListHashSet to keep the order of the parameters.
    for (auto& parameter : extensionParameters) {
        m_acceptedExtensionsBuilder.append("; "_s, parameter.key);
        if (!parameter.value.isNull())
            m_acceptedExtensionsBuilder.append('=', parameter.value);
    }
}

void WebSocketExtensionDispatcher::fail(const String& reason)
{
    m_failureReason = reason;
    m_acceptedExtensionsBuilder.clear();
}

bool WebSocketExtensionDispatcher::processHeaderValue(const String& headerValue)
{
    if (!headerValue.length())
        return true;

    // If we don't send Sec-WebSocket-Extensions header, the server should not return the header.
    if (!m_processors.size()) {
        fail("Received unexpected Sec-WebSocket-Extensions header"_s);
        return false;
    }

    const CString headerValueData = headerValue.utf8();
    WebSocketExtensionParser parser(headerValueData.span());
    while (!parser.finished()) {
        String extensionToken;
        HashMap<String, String> extensionParameters;
        if (!parser.parseExtension(extensionToken, extensionParameters)) {
            fail("Sec-WebSocket-Extensions header is invalid"_s);
            return false;
        }

        size_t index = 0;
        for (auto& processor : m_processors) {
            if (extensionToken == processor->extensionToken()) {
                if (processor->processResponse(extensionParameters)) {
                    appendAcceptedExtension(extensionToken, extensionParameters);
                    break;
                }
                fail(processor->failureReason());
                return false;
            }
            ++index;
        }
        // There is no extension which can process the response.
        if (index == m_processors.size()) {
            fail(makeString("Received unexpected extension: "_s, extensionToken));
            return false;
        }
    }
    return parser.parsedSuccessfully();
}

String WebSocketExtensionDispatcher::acceptedExtensions() const
{
    if (m_acceptedExtensionsBuilder.isEmpty())
        return String();
    return m_acceptedExtensionsBuilder.toStringPreserveCapacity();
}

String WebSocketExtensionDispatcher::failureReason() const
{
    return m_failureReason;
}

} // namespace WebCore
