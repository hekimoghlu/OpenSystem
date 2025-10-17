/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 4, 2022.
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

#include "APIClient.h"
#include "APIDiagnosticLoggingClient.h"
#include "WKPage.h"
#include <wtf/Forward.h>

namespace API {

template<> struct ClientTraits<WKPageDiagnosticLoggingClientBase> {
    typedef std::tuple<WKPageDiagnosticLoggingClientV0, WKPageDiagnosticLoggingClientV1> Versions;
};

} // namespace API

namespace WebKit {

class WebPageProxy;

class WebPageDiagnosticLoggingClient final : public API::Client<WKPageDiagnosticLoggingClientBase>, public API::DiagnosticLoggingClient {
public:
    explicit WebPageDiagnosticLoggingClient(const WKPageDiagnosticLoggingClientBase*);

    void logDiagnosticMessage(WebPageProxy*, const String& message, const String& description) override;
    void logDiagnosticMessageWithResult(WebPageProxy*, const String& message, const String& description, WebCore::DiagnosticLoggingResultType) override;
    void logDiagnosticMessageWithValue(WebPageProxy*, const String& message, const String& description, const String& value) override;
    void logDiagnosticMessageWithEnhancedPrivacy(WebPageProxy*, const String& message, const String& description) override;
    void logDiagnosticMessageWithValueDictionary(WebKit::WebPageProxy*, const WTF::String& message, const WTF::String& description, Ref<API::Dictionary>&&) override { }
};

} // namespace WebKit
