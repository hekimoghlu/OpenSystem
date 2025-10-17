/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 14, 2025.
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

#include <wtf/Forward.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {
enum DiagnosticLoggingResultType : uint8_t;
enum class DiagnosticLoggingDomain : uint8_t;
}

namespace WebKit {
class WebPageProxy;
}

namespace API {

class Dictionary;

class DiagnosticLoggingClient {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(DiagnosticLoggingClient);
public:
    virtual ~DiagnosticLoggingClient() { }

    virtual void logDiagnosticMessage(WebKit::WebPageProxy*, const WTF::String& message, const WTF::String& description) = 0;
    virtual void logDiagnosticMessageWithResult(WebKit::WebPageProxy*, const WTF::String& message, const WTF::String& description, WebCore::DiagnosticLoggingResultType) = 0;
    virtual void logDiagnosticMessageWithValue(WebKit::WebPageProxy*, const WTF::String& message, const WTF::String& description, const WTF::String& value) = 0;
    virtual void logDiagnosticMessageWithEnhancedPrivacy(WebKit::WebPageProxy*, const WTF::String& message, const WTF::String& description) = 0;

    virtual void logDiagnosticMessageWithValueDictionary(WebKit::WebPageProxy*, const WTF::String& message, const WTF::String& description, Ref<API::Dictionary>&&) = 0;

    virtual void logDiagnosticMessageWithDomain(WebKit::WebPageProxy*, const WTF::String&, WebCore::DiagnosticLoggingDomain) { }
};

} // namespace API
