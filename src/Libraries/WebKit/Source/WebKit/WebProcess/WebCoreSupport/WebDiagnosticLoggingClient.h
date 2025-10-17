/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 10, 2022.
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
#ifndef WebDiagnosticLoggingClient_h
#define WebDiagnosticLoggingClient_h

#include <WebCore/DiagnosticLoggingClient.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakRef.h>

namespace WebKit {

class WebPage;

class WebDiagnosticLoggingClient : public WebCore::DiagnosticLoggingClient {
    WTF_MAKE_TZONE_ALLOCATED(WebDiagnosticLoggingClient);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(WebDiagnosticLoggingClient);
public:
    WebDiagnosticLoggingClient(WebPage&);
    virtual ~WebDiagnosticLoggingClient();

private:
    void logDiagnosticMessage(const String& message, const String& description, WebCore::ShouldSample) override;
    void logDiagnosticMessageWithResult(const String& message, const String& description, WebCore::DiagnosticLoggingResultType, WebCore::ShouldSample) override;
    void logDiagnosticMessageWithValue(const String& message, const String& description, double value, unsigned significantFigures, WebCore::ShouldSample) override;
    void logDiagnosticMessageWithEnhancedPrivacy(const String& message, const String& description, WebCore::ShouldSample) override;
    void logDiagnosticMessageWithValueDictionary(const String& message, const String& description, const ValueDictionary&, WebCore::ShouldSample) override;
    void logDiagnosticMessageWithDomain(const String& message, WebCore::DiagnosticLoggingDomain) override;

    Ref<WebPage> protectedPage() const;

    WeakRef<WebPage> m_page;
};

}

#endif
