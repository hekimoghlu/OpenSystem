/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 18, 2025.
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
#import "DiagnosticLoggingClient.h"

#import "APIDictionary.h"
#import "WKSharedAPICast.h"
#import "_WKDiagnosticLoggingDelegate.h"
#import <WebCore/DiagnosticLoggingDomain.h>
#import <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(DiagnosticLoggingClient);

DiagnosticLoggingClient::DiagnosticLoggingClient(WKWebView *webView)
    : m_webView(webView)
{
}

RetainPtr<id <_WKDiagnosticLoggingDelegate>> DiagnosticLoggingClient::delegate()
{
    return m_delegate.get();
}

void DiagnosticLoggingClient::setDelegate(id <_WKDiagnosticLoggingDelegate> delegate)
{
    m_delegate = delegate;

    m_delegateMethods.webviewLogDiagnosticMessage = [delegate respondsToSelector:@selector(_webView:logDiagnosticMessage:description:)];
    m_delegateMethods.webviewLogDiagnosticMessageWithResult = [delegate respondsToSelector:@selector(_webView:logDiagnosticMessageWithResult:description:result:)];
    m_delegateMethods.webviewLogDiagnosticMessageWithValue = [delegate respondsToSelector:@selector(_webView:logDiagnosticMessageWithValue:description:value:)];
    m_delegateMethods.webviewLogDiagnosticMessageWithEnhancedPrivacy = [delegate respondsToSelector:@selector(_webView:logDiagnosticMessageWithEnhancedPrivacy:description:)];
    m_delegateMethods.webviewLogDiagnosticMessageWithValueDictionary = [delegate respondsToSelector:@selector(_webView:logDiagnosticMessage:description:valueDictionary:)];
    m_delegateMethods.webviewLogDiagnosticMessageWithDomain = [delegate respondsToSelector:@selector(_webView:logDiagnosticMessageWithDomain:domain:)];
}

void DiagnosticLoggingClient::logDiagnosticMessage(WebKit::WebPageProxy*, const WTF::String& message, const WTF::String& description)
{
    if (m_delegateMethods.webviewLogDiagnosticMessage)
        [m_delegate.get() _webView:m_webView logDiagnosticMessage:message description:description];
}

static _WKDiagnosticLoggingResultType toWKDiagnosticLoggingResultType(WebCore::DiagnosticLoggingResultType result)
{
    switch (result) {
    case WebCore::DiagnosticLoggingResultPass:
        return _WKDiagnosticLoggingResultPass;
    case WebCore::DiagnosticLoggingResultFail:
        return _WKDiagnosticLoggingResultFail;
    case WebCore::DiagnosticLoggingResultNoop:
        return _WKDiagnosticLoggingResultNoop;
    }
}

static _WKDiagnosticLoggingDomain toWKDiagnosticLoggingDomain(WebCore::DiagnosticLoggingDomain domain)
{
    switch (domain) {
    case WebCore::DiagnosticLoggingDomain::Media:
        return _WKDiagnosticLoggingDomainMedia;
    }
}

void DiagnosticLoggingClient::logDiagnosticMessageWithResult(WebKit::WebPageProxy*, const WTF::String& message, const WTF::String& description, WebCore::DiagnosticLoggingResultType result)
{
    if (m_delegateMethods.webviewLogDiagnosticMessageWithResult)
        [m_delegate.get() _webView:m_webView logDiagnosticMessageWithResult:message description:description result:toWKDiagnosticLoggingResultType(result)];
}

void DiagnosticLoggingClient::logDiagnosticMessageWithValue(WebKit::WebPageProxy*, const WTF::String& message, const WTF::String& description, const WTF::String& value)
{
    if (m_delegateMethods.webviewLogDiagnosticMessageWithValue)
        [m_delegate.get() _webView:m_webView logDiagnosticMessageWithValue:message description:description value:value];
}

void DiagnosticLoggingClient::logDiagnosticMessageWithEnhancedPrivacy(WebKit::WebPageProxy*, const WTF::String& message, const WTF::String& description)
{
    if (m_delegateMethods.webviewLogDiagnosticMessageWithEnhancedPrivacy)
        [m_delegate.get() _webView:m_webView logDiagnosticMessageWithEnhancedPrivacy:message description:description];
}

void DiagnosticLoggingClient::logDiagnosticMessageWithValueDictionary(WebPageProxy*, const String& message, const String& description, Ref<API::Dictionary>&& valueDictionary)
{
    if (m_delegateMethods.webviewLogDiagnosticMessageWithValueDictionary)
        [m_delegate.get() _webView:m_webView logDiagnosticMessage:message description:description valueDictionary:static_cast<NSDictionary*>(valueDictionary->wrapper())];
}

void DiagnosticLoggingClient::logDiagnosticMessageWithDomain(WebPageProxy*, const String& message, WebCore::DiagnosticLoggingDomain domain)
{
    if (m_delegateMethods.webviewLogDiagnosticMessageWithDomain)
        [m_delegate.get() _webView:m_webView logDiagnosticMessageWithDomain:message domain:toWKDiagnosticLoggingDomain(domain)];
}

} // namespace WebKit
