/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 1, 2021.
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
#include "WebPageDiagnosticLoggingClient.h"

#include "WKAPICast.h"
#include "WebPageProxy.h"
#include <wtf/text/WTFString.h>

namespace WebKit {

WebPageDiagnosticLoggingClient::WebPageDiagnosticLoggingClient(const WKPageDiagnosticLoggingClientBase* client)
{
    initialize(client);
}

void WebPageDiagnosticLoggingClient::logDiagnosticMessage(WebPageProxy* page, const String& message, const String& description)
{
    if (!m_client.logDiagnosticMessage)
        return;

    m_client.logDiagnosticMessage(toAPI(page), toAPI(message.impl()), toAPI(description.impl()), m_client.base.clientInfo);
}

void WebPageDiagnosticLoggingClient::logDiagnosticMessageWithResult(WebPageProxy* page, const String& message, const String& description, WebCore::DiagnosticLoggingResultType result)
{
    if (!m_client.logDiagnosticMessageWithResult)
        return;

    m_client.logDiagnosticMessageWithResult(toAPI(page), toAPI(message.impl()), toAPI(description.impl()), toAPI(result), m_client.base.clientInfo);
}

void WebPageDiagnosticLoggingClient::logDiagnosticMessageWithValue(WebPageProxy* page, const String& message, const String& description, const String& value)
{
    if (!m_client.logDiagnosticMessageWithValue)
        return;

    m_client.logDiagnosticMessageWithValue(toAPI(page), toAPI(message.impl()), toAPI(description.impl()), toAPI(value.impl()), m_client.base.clientInfo);
}

void WebPageDiagnosticLoggingClient::logDiagnosticMessageWithEnhancedPrivacy(WebPageProxy* page, const String& message, const String& description)
{
    if (!m_client.logDiagnosticMessageWithEnhancedPrivacy)
        return;

    m_client.logDiagnosticMessageWithEnhancedPrivacy(toAPI(page), toAPI(message.impl()), toAPI(description.impl()), m_client.base.clientInfo);
}

} // namespace WebKit
