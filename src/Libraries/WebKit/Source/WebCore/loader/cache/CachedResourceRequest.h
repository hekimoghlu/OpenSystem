/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 31, 2024.
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

#include "CachedResource.h"
#include "ContentSecurityPolicy.h"
#include "Element.h"
#include "ResourceLoadPriority.h"
#include "ResourceLoaderOptions.h"
#include "ResourceRequest.h"
#include "SecurityOrigin.h"
#include "ServiceWorkerIdentifier.h"
#include <wtf/RefPtr.h>
#include <wtf/text/AtomString.h>

namespace WebCore {

struct ContentRuleListResults;
class Document;
class FrameLoader;
class Page;
struct ServiceWorkerRegistrationData;
enum class CachePolicy : uint8_t;
enum class ReferrerPolicy : uint8_t;

bool isRequestCrossOrigin(SecurityOrigin*, const URL& requestURL, const ResourceLoaderOptions&);

class CachedResourceRequest {
public:
    CachedResourceRequest(ResourceRequest&&, const ResourceLoaderOptions&, std::optional<ResourceLoadPriority> = std::nullopt, String&& charset = String());

    ResourceRequest&& releaseResourceRequest() { return WTFMove(m_resourceRequest); }
    const ResourceRequest& resourceRequest() const { return m_resourceRequest; }
    ResourceRequest& resourceRequest() { return m_resourceRequest; }

    const String& charset() const { return m_charset; }
    void setCharset(const String& charset) { m_charset = charset; }

    const ResourceLoaderOptions& options() const { return m_options; }
    void setOptions(const ResourceLoaderOptions& options) { m_options = options; }

    const std::optional<ResourceLoadPriority>& priority() const { return m_priority; }
    void setPriority(std::optional<ResourceLoadPriority>&& priority) { m_priority = WTFMove(priority); }

    RequestPriority fetchPriority() const { return m_options.fetchPriority; }

    void setInitiator(Element&);
    void setInitiatorType(const AtomString&);
    const AtomString& initiatorType() const;

    bool allowsCaching() const { return m_options.cachingPolicy == CachingPolicy::AllowCaching; }
    void setCachingPolicy(CachingPolicy policy) { m_options.cachingPolicy = policy;  }

    // Whether this request should impact request counting and delay window.onload.
    bool ignoreForRequestCount() const { return m_ignoreForRequestCount; }
    void setIgnoreForRequestCount(bool ignoreForRequestCount) { m_ignoreForRequestCount = ignoreForRequestCount; }

    void setDestinationIfNotSet(FetchOptions::Destination);

    void updateForAccessControl(Document&);

    void updateReferrerPolicy(ReferrerPolicy);
    void updateReferrerAndOriginHeaders(FrameLoader&);
    void updateUserAgentHeader(FrameLoader&);
    void upgradeInsecureRequestIfNeeded(Document&, ContentSecurityPolicy::AlwaysUpgradeRequest = ContentSecurityPolicy::AlwaysUpgradeRequest::No);
    void setAcceptHeaderIfNone(CachedResource::Type);
    void updateAccordingCacheMode();
    void updateAcceptEncodingHeader();
    void updateCacheModeIfNeeded(CachePolicy);

    void disableCachingIfNeeded();

    void removeFragmentIdentifierIfNeeded();
#if ENABLE(CONTENT_EXTENSIONS)
    void applyResults(ContentRuleListResults&&, Page*);
#endif
    void setDomainForCachePartition(Document&);
    void setDomainForCachePartition(const String&);
    bool isLinkPreload() const { return m_isLinkPreload; }
    void setIsLinkPreload() { m_isLinkPreload = true; }

    void setOrigin(Ref<SecurityOrigin>&& origin) { m_origin = WTFMove(origin); }
    RefPtr<SecurityOrigin> releaseOrigin() { return WTFMove(m_origin); }
    const SecurityOrigin* origin() const { return m_origin.get(); }
    SecurityOrigin* origin() { return m_origin.get(); }
    RefPtr<SecurityOrigin> protectedOrigin() const { return m_origin; }

    bool hasFragmentIdentifier() const { return !m_fragmentIdentifier.isEmpty(); }
    String&& releaseFragmentIdentifier() { return WTFMove(m_fragmentIdentifier); }
    void clearFragmentIdentifier() { m_fragmentIdentifier = { }; }

    static String splitFragmentIdentifierFromRequestURL(ResourceRequest&);
    static String acceptHeaderValueFromType(CachedResource::Type);

    void setClientIdentifierIfNeeded(ScriptExecutionContextIdentifier);
    void setSelectedServiceWorkerRegistrationIdentifierIfNeeded(ServiceWorkerRegistrationIdentifier);
    void setNavigationServiceWorkerRegistrationData(const std::optional<ServiceWorkerRegistrationData>&);

private:
    ResourceRequest m_resourceRequest;
    String m_charset;
    ResourceLoaderOptions m_options;
    std::optional<ResourceLoadPriority> m_priority;
    RefPtr<Element> m_initiatorElement;
    AtomString m_initiatorType;
    RefPtr<SecurityOrigin> m_origin;
    String m_fragmentIdentifier;
    bool m_isLinkPreload { false };
    bool m_ignoreForRequestCount { false };
};

void upgradeInsecureResourceRequestIfNeeded(ResourceRequest&, Document&, ContentSecurityPolicy::AlwaysUpgradeRequest = ContentSecurityPolicy::AlwaysUpgradeRequest::No);

} // namespace WebCore
