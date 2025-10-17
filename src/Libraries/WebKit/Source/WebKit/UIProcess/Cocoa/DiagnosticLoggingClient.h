/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 17, 2023.
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

#import "APIDiagnosticLoggingClient.h"
#import "WKFoundation.h"
#import <wtf/TZoneMalloc.h>
#import <wtf/WeakObjCPtr.h>

@class WKWebView;
@protocol _WKDiagnosticLoggingDelegate;

namespace WebKit {

class DiagnosticLoggingClient final : public API::DiagnosticLoggingClient {
    WTF_MAKE_TZONE_ALLOCATED(DiagnosticLoggingClient);
public:
    explicit DiagnosticLoggingClient(WKWebView *);

    RetainPtr<id <_WKDiagnosticLoggingDelegate>> delegate();
    void setDelegate(id <_WKDiagnosticLoggingDelegate>);

private:
    // From API::DiagnosticLoggingClient
    void logDiagnosticMessage(WebPageProxy*, const String& message, const String& description) override;
    void logDiagnosticMessageWithResult(WebPageProxy*, const String& message, const String& description, WebCore::DiagnosticLoggingResultType) override;
    void logDiagnosticMessageWithValue(WebPageProxy*, const String& message, const String& description, const String& value) override;
    void logDiagnosticMessageWithEnhancedPrivacy(WebPageProxy*, const String& message, const String& description) override;
    void logDiagnosticMessageWithValueDictionary(WebPageProxy*, const String& message, const String& description, Ref<API::Dictionary>&&) override;
    void logDiagnosticMessageWithDomain(WebPageProxy*, const String& message, WebCore::DiagnosticLoggingDomain) override;

    WKWebView *m_webView;
    WeakObjCPtr<id <_WKDiagnosticLoggingDelegate>> m_delegate;

    struct {
        unsigned webviewLogDiagnosticMessage : 1;
        unsigned webviewLogDiagnosticMessageWithResult : 1;
        unsigned webviewLogDiagnosticMessageWithValue : 1;
        unsigned webviewLogDiagnosticMessageWithEnhancedPrivacy : 1;
        unsigned webviewLogDiagnosticMessageWithValueDictionary : 1;
        unsigned webviewLogDiagnosticMessageWithDomain : 1;
    } m_delegateMethods;
};

} // WebKit
