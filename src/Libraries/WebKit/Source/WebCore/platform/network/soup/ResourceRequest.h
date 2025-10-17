/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 8, 2025.
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

#include "PageIdentifier.h"
#include "ResourceRequestBase.h"
#include "URLSoup.h"
#include <wtf/glib/GRefPtr.h>

namespace WebCore {

class BlobRegistryImpl;

struct ResourceRequestPlatformData {
    ResourceRequestBase::RequestData requestData;
    bool acceptEncoding;
    uint16_t redirectCount;
};
using ResourceRequestData = std::variant<ResourceRequestBase::RequestData, ResourceRequestPlatformData>;

class ResourceRequest : public ResourceRequestBase {
public:
    explicit ResourceRequest(const String& url)
        : ResourceRequestBase(URL({ }, url), ResourceRequestCachePolicy::UseProtocolCachePolicy)
    {
    }

    ResourceRequest(const URL& url)
        : ResourceRequestBase(url, ResourceRequestCachePolicy::UseProtocolCachePolicy)
    {
    }

    ResourceRequest(const URL& url, const String& referrer, ResourceRequestCachePolicy policy = ResourceRequestCachePolicy::UseProtocolCachePolicy)
        : ResourceRequestBase(url, policy)
    {
        setHTTPReferrer(referrer);
    }

    ResourceRequest()
        : ResourceRequestBase(URL(), ResourceRequestCachePolicy::UseProtocolCachePolicy)
    {
    }

    ResourceRequest(ResourceRequestBase&& base)
        : ResourceRequestBase(WTFMove(base))
    {
    }

    explicit ResourceRequest(ResourceRequestPlatformData&& platformData)
        : ResourceRequestBase(WTFMove(platformData.requestData))
        , m_acceptEncoding(platformData.acceptEncoding)
        , m_redirectCount(platformData.redirectCount)
    {
    }

    GRefPtr<SoupMessage> createSoupMessage(BlobRegistryImpl&) const;
    GRefPtr<GInputStream> createBodyStream() const;

    void updateFromDelegatePreservingOldProperties(const ResourceRequest& delegateProvidedRequest);

    bool acceptEncoding() const { return m_acceptEncoding; }
    void setAcceptEncoding(bool acceptEncoding) { m_acceptEncoding = acceptEncoding; }

    void incrementRedirectCount() { m_redirectCount++; }
    uint16_t redirectCount() const { return m_redirectCount; }

    void updateSoupMessageBody(SoupMessage*, BlobRegistryImpl&) const;
    void updateSoupMessageHeaders(SoupMessageHeaders*) const;
    void updateFromSoupMessageHeaders(SoupMessageHeaders*);

    // We only need to encode platform data if acceptEncoding or redirectCount are not the default.
    bool encodingRequiresPlatformData() const { return !m_acceptEncoding || m_redirectCount; }

    WEBCORE_EXPORT static ResourceRequest fromResourceRequestData(ResourceRequestData);
    WEBCORE_EXPORT ResourceRequestData getRequestDataToSerialize() const;

private:
    friend class ResourceRequestBase;

#if USE(SOUP2)
    GUniquePtr<SoupURI> createSoupURI() const;
#else
    GRefPtr<GUri> createSoupURI() const;
#endif

    void doUpdatePlatformRequest() { }
    void doUpdateResourceRequest() { }
    void doUpdatePlatformHTTPBody() { }
    void doUpdateResourceHTTPBody() { }

    void doPlatformSetAsIsolatedCopy(const ResourceRequest&) { }

    bool m_acceptEncoding { true };
    uint16_t m_redirectCount { 0 };
};

} // namespace WebCore

