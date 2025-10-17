/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 21, 2025.
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

#include "CertificateInfo.h"
#include "NetworkLoadMetrics.h"
#include <wtf/URL.h>

namespace WebCore {

class CurlResponse {
public:
    CurlResponse() = default;

    CurlResponse isolatedCopy() const
    {
        CurlResponse copy;

        copy.url = url.isolatedCopy();
        copy.statusCode = statusCode;
        copy.httpConnectCode = httpConnectCode;
        copy.expectedContentLength = expectedContentLength;

        for (const auto& header : headers)
            copy.headers.append(header.isolatedCopy());

        copy.proxyUrl = proxyUrl.isolatedCopy();
        copy.availableHttpAuth = availableHttpAuth;
        copy.availableProxyAuth = availableProxyAuth;
        copy.httpVersion = httpVersion;

        copy.certificateInfo = certificateInfo.isolatedCopy();
        copy.networkLoadMetrics = networkLoadMetrics.isolatedCopy();

        return copy;
    }

    URL url;
    long statusCode { 0 };
    long httpConnectCode { 0 };
    long long expectedContentLength { 0 };
    Vector<String> headers;

    URL proxyUrl;
    long availableHttpAuth { 0 };
    long availableProxyAuth { 0 };
    long httpVersion { 0 };

    CertificateInfo certificateInfo;
    NetworkLoadMetrics networkLoadMetrics;
};

} // namespace WebCore
