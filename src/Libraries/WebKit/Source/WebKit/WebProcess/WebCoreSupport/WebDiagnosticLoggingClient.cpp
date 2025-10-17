/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 8, 2024.
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
#include "WebDiagnosticLoggingClient.h"

#include "MessageSenderInlines.h"
#include "WebPage.h"
#include "WebPageProxyMessages.h"
#include <WebCore/Page.h>
#include <WebCore/Settings.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {
using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebDiagnosticLoggingClient);

WebDiagnosticLoggingClient::WebDiagnosticLoggingClient(WebPage& page)
    : m_page(page)
{
}

WebDiagnosticLoggingClient::~WebDiagnosticLoggingClient() = default;

Ref<WebPage> WebDiagnosticLoggingClient::protectedPage() const
{
    return m_page.get();
}

void WebDiagnosticLoggingClient::logDiagnosticMessage(const String& message, const String& description, WebCore::ShouldSample shouldSample)
{
    ASSERT(!m_page->corePage() || m_page->corePage()->settings().diagnosticLoggingEnabled());

    if (!shouldLogAfterSampling(shouldSample))
        return;

    protectedPage()->send(Messages::WebPageProxy::LogDiagnosticMessageFromWebProcess(message, description, ShouldSample::No));
}

void WebDiagnosticLoggingClient::logDiagnosticMessageWithResult(const String& message, const String& description, WebCore::DiagnosticLoggingResultType result, WebCore::ShouldSample shouldSample)
{
    ASSERT(!m_page->corePage() || m_page->corePage()->settings().diagnosticLoggingEnabled());

    if (!shouldLogAfterSampling(shouldSample))
        return;

    protectedPage()->send(Messages::WebPageProxy::LogDiagnosticMessageWithResultFromWebProcess(message, description, result, ShouldSample::No));
}

void WebDiagnosticLoggingClient::logDiagnosticMessageWithValue(const String& message, const String& description, double value, unsigned significantFigures, WebCore::ShouldSample shouldSample)
{
    ASSERT(!m_page->corePage() || m_page->corePage()->settings().diagnosticLoggingEnabled());

    if (!shouldLogAfterSampling(shouldSample))
        return;

    protectedPage()->send(Messages::WebPageProxy::LogDiagnosticMessageWithValueFromWebProcess(message, description, value, significantFigures, ShouldSample::No));
}

void WebDiagnosticLoggingClient::logDiagnosticMessageWithEnhancedPrivacy(const String& message, const String& description, WebCore::ShouldSample shouldSample)
{
    ASSERT(!m_page->corePage() || m_page->corePage()->settings().diagnosticLoggingEnabled());

    if (!shouldLogAfterSampling(shouldSample))
        return;

    protectedPage()->send(Messages::WebPageProxy::LogDiagnosticMessageWithEnhancedPrivacyFromWebProcess(message, description, ShouldSample::No));
}

void WebDiagnosticLoggingClient::logDiagnosticMessageWithValueDictionary(const String& message, const String& description, const ValueDictionary& value, ShouldSample shouldSample)
{
    ASSERT(!m_page->corePage() || m_page->corePage()->settings().diagnosticLoggingEnabled());

    if (!shouldLogAfterSampling(shouldSample))
        return;

    protectedPage()->send(Messages::WebPageProxy::LogDiagnosticMessageWithValueDictionaryFromWebProcess(message, description, value, ShouldSample::No));
}

void WebDiagnosticLoggingClient::logDiagnosticMessageWithDomain(const String& message, WebCore::DiagnosticLoggingDomain domain)
{
    ASSERT(!m_page->corePage() || m_page->corePage()->settings().diagnosticLoggingEnabled());

    protectedPage()->send(Messages::WebPageProxy::LogDiagnosticMessageWithDomainFromWebProcess(message, domain));
}

} // namespace WebKit
