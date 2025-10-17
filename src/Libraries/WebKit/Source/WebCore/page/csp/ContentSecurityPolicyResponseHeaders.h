/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 6, 2025.
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

#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class ContentSecurityPolicy;
class ResourceResponse;

enum class ContentSecurityPolicyHeaderType : bool {
    Report,
    Enforce,
};

class ContentSecurityPolicyResponseHeaders {
public:
    ContentSecurityPolicyResponseHeaders() = default;
    ContentSecurityPolicyResponseHeaders(Vector<std::pair<String, ContentSecurityPolicyHeaderType>>&& headers, int httpStatusCode)
        : m_headers(WTFMove(headers))
        , m_httpStatusCode(httpStatusCode)
    { }

    WEBCORE_EXPORT explicit ContentSecurityPolicyResponseHeaders(const ResourceResponse&);

    ContentSecurityPolicyResponseHeaders isolatedCopy() const &;
    ContentSecurityPolicyResponseHeaders isolatedCopy() &&;

    enum EmptyTag { Empty };
    struct MarkableTraits {
        static bool isEmptyValue(const ContentSecurityPolicyResponseHeaders& identifier)
        {
            return identifier.m_emptyForMarkable;
        }

        static ContentSecurityPolicyResponseHeaders emptyValue()
        {
            return ContentSecurityPolicyResponseHeaders(Empty);
        }
    };

    void addPolicyHeadersTo(ResourceResponse&) const;

    const Vector<std::pair<String, ContentSecurityPolicyHeaderType>>& headers() const { return m_headers; }
    void setHeaders(Vector<std::pair<String, ContentSecurityPolicyHeaderType>>&& headers) { m_headers = WTFMove(headers); }
    int httpStatusCode() const { return m_httpStatusCode; }
    void setHTTPStatusCode(int httpStatusCode) { m_httpStatusCode = httpStatusCode; }

private:
    friend bool operator==(const ContentSecurityPolicyResponseHeaders&, const ContentSecurityPolicyResponseHeaders&);
    friend class ContentSecurityPolicy;
    ContentSecurityPolicyResponseHeaders(EmptyTag)
        : m_emptyForMarkable(true)
    { }

    Vector<std::pair<String, ContentSecurityPolicyHeaderType>> m_headers;
    int m_httpStatusCode { 0 };
    bool m_emptyForMarkable { false };
};

inline bool operator==(const ContentSecurityPolicyResponseHeaders&a, const ContentSecurityPolicyResponseHeaders&b)
{
    return a.m_headers == b.m_headers;
}

} // namespace WebCore
