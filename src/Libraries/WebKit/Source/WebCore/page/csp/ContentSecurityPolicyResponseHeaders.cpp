/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 1, 2024.
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
#include "ContentSecurityPolicyResponseHeaders.h"

#include "HTTPHeaderNames.h"
#include "ResourceResponse.h"
#include <wtf/CrossThreadCopier.h>

namespace WebCore {

ContentSecurityPolicyResponseHeaders::ContentSecurityPolicyResponseHeaders(const ResourceResponse& response)
{
    String policyValue = response.httpHeaderField(HTTPHeaderName::ContentSecurityPolicy);
    if (!policyValue.isEmpty())
        m_headers.append({ policyValue, ContentSecurityPolicyHeaderType::Enforce });

    policyValue = response.httpHeaderField(HTTPHeaderName::ContentSecurityPolicyReportOnly);
    if (!policyValue.isEmpty())
        m_headers.append({ policyValue, ContentSecurityPolicyHeaderType::Report });

    m_httpStatusCode = response.httpStatusCode();
}

ContentSecurityPolicyResponseHeaders ContentSecurityPolicyResponseHeaders::isolatedCopy() const &
{
    ContentSecurityPolicyResponseHeaders isolatedCopy;
    isolatedCopy.m_headers = crossThreadCopy(m_headers);
    isolatedCopy.m_httpStatusCode = m_httpStatusCode;
    return isolatedCopy;
}

ContentSecurityPolicyResponseHeaders ContentSecurityPolicyResponseHeaders::isolatedCopy() &&
{
    ContentSecurityPolicyResponseHeaders isolatedCopy;
    isolatedCopy.m_headers = crossThreadCopy(WTFMove(m_headers));
    isolatedCopy.m_httpStatusCode = m_httpStatusCode;
    return isolatedCopy;
}

void ContentSecurityPolicyResponseHeaders::addPolicyHeadersTo(ResourceResponse& response) const
{
    for (const auto& header : m_headers) {
        switch (header.second) {
        case ContentSecurityPolicyHeaderType::Enforce:
            response.setHTTPHeaderField(HTTPHeaderName::ContentSecurityPolicy, header.first);
            break;
        case ContentSecurityPolicyHeaderType::Report:
            response.setHTTPHeaderField(HTTPHeaderName::ContentSecurityPolicyReportOnly, header.first);
            break;
        }
    }
}

} // namespace WebCore
