/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 3, 2021.
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

#include "ResourceRequestBase.h"
#include <wtf/RetainPtr.h>
#include <wtf/TZoneMalloc.h>

OBJC_CLASS NSCachedURLResponse;
OBJC_CLASS NSURLRequest;

typedef const struct _CFURLRequest* CFURLRequestRef;
typedef const struct __CFURLStorageSession* CFURLStorageSessionRef;

namespace WebCore {

struct ResourceRequestPlatformData {
    RetainPtr<NSURLRequest> m_urlRequest;
    std::optional<bool> m_isAppInitiated;
    std::optional<ResourceRequestRequester> m_requester;
    bool m_privacyProxyFailClosedForUnreachableNonMainHosts { false };
    bool m_useAdvancedPrivacyProtections { false };
    bool m_didFilterLinkDecoration { false };
    bool m_isPrivateTokenUsageByThirdPartyAllowed { false };
    bool m_wasSchemeOptimisticallyUpgraded { false };
};

using ResourceRequestData = std::variant<ResourceRequestBase::RequestData, ResourceRequestPlatformData>;

class ResourceRequest : public ResourceRequestBase {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(ResourceRequest, WEBCORE_EXPORT);
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
    
    WEBCORE_EXPORT ResourceRequest(NSURLRequest *);

    ResourceRequest(ResourceRequestBase&& base
        , const String& cachePartition
        , bool hiddenFromInspector
    )
        : ResourceRequestBase(WTFMove(base))
    {
        m_cachePartition = cachePartition;
        m_hiddenFromInspector = hiddenFromInspector;
    }

    ResourceRequest(ResourceRequestPlatformData&&, const String& cachePartition, bool hiddenFromInspector);

    WEBCORE_EXPORT static ResourceRequest fromResourceRequestData(ResourceRequestData, const String& cachePartition, bool hiddenFromInspector);

    WEBCORE_EXPORT void updateFromDelegatePreservingOldProperties(const ResourceRequest&);
    
    bool encodingRequiresPlatformData() const { return m_httpBody || m_nsRequest; }
    WEBCORE_EXPORT NSURLRequest *nsURLRequest(HTTPBodyUpdatePolicy) const;

    WEBCORE_EXPORT static CFStringRef isUserInitiatedKey();
    WEBCORE_EXPORT ResourceRequestPlatformData getResourceRequestPlatformData() const;
    WEBCORE_EXPORT ResourceRequestData getRequestDataToSerialize() const;
    WEBCORE_EXPORT CFURLRequestRef cfURLRequest(HTTPBodyUpdatePolicy) const;
    void setStorageSession(CFURLStorageSessionRef);

    WEBCORE_EXPORT static bool httpPipeliningEnabled();
    WEBCORE_EXPORT static void setHTTPPipeliningEnabled(bool);

    static bool resourcePrioritiesEnabled() { return true; }

    WEBCORE_EXPORT void replacePlatformRequest(HTTPBodyUpdatePolicy);

private:
    friend class ResourceRequestBase;

    void doUpdatePlatformRequest();
    void doUpdateResourceRequest();
    void doUpdatePlatformHTTPBody();
    void doUpdateResourceHTTPBody();

    void doPlatformSetAsIsolatedCopy(const ResourceRequest&);

    RetainPtr<NSURLRequest> m_nsRequest;
    static bool s_httpPipeliningEnabled;
};

RetainPtr<NSURLRequest> copyRequestWithStorageSession(CFURLStorageSessionRef, NSURLRequest *);
WEBCORE_EXPORT NSCachedURLResponse *cachedResponseForRequest(CFURLStorageSessionRef, NSURLRequest *);

} // namespace WebCore
