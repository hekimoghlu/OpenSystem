/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 27, 2024.
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

#include "ResourceResponseBase.h"

namespace WebCore {

class CurlResponse;

class ResourceResponse : public ResourceResponseBase {
public:
    ResourceResponse()
        : ResourceResponseBase()
    {
    }

    ResourceResponse(const URL& url, const String& mimeType, long long expectedLength, const String& textEncodingName)
        : ResourceResponseBase(url, mimeType, expectedLength, textEncodingName)
    {
    }

    ResourceResponse(ResourceResponseBase&& base)
        : ResourceResponseBase(WTFMove(base))
    {
    }

    WEBCORE_EXPORT ResourceResponse(CurlResponse&);

    bool isMovedPermanently() const { return httpStatusCode() == 301; };
    bool isFound() const { return httpStatusCode() == 302; }
    bool isSeeOther() const { return httpStatusCode() == 303; }
    bool isUnauthorized() const { return httpStatusCode() == 401; }
    bool isProxyAuthenticationRequired() const { return httpStatusCode() == 407; }

private:
    friend class ResourceResponseBase;

    String platformSuggestedFilename() const;

    void appendHTTPHeaderField(const String&);
};

} // namespace WebCore
