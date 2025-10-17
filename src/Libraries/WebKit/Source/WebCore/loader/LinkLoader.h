/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 22, 2022.
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
#include "CachedResourceClient.h"
#include "CachedResourceHandle.h"
#include "LinkLoaderClient.h"
#include "LinkRelAttribute.h"
#include "ReferrerPolicy.h"

namespace WebCore {
class LinkLoader;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::LinkLoader> : std::true_type { };
}

namespace WebCore {

class Document;
class LinkPreloadResourceClient;

struct LinkLoadParameters {
    LinkRelAttribute relAttribute;
    URL href;
    String as;
    String media;
    String mimeType;
    String crossOrigin;
    String imageSrcSet;
    String imageSizes;
    String nonce;
    ReferrerPolicy referrerPolicy { ReferrerPolicy::EmptyString };
    RequestPriority fetchPriority { RequestPriority::Auto };
};

class LinkLoader : public CachedResourceClient {
public:
    explicit LinkLoader(LinkLoaderClient&);
    virtual ~LinkLoader();

    void loadLink(const LinkLoadParameters&, Document&);
    enum class ShouldLog : bool { No, Yes };
    static std::optional<CachedResource::Type> resourceTypeFromAsAttribute(const String&, Document&, ShouldLog = ShouldLog::No);

    enum class MediaAttributeCheck { MediaAttributeEmpty, MediaAttributeNotEmpty, SkipMediaAttributeCheck };
    static void loadLinksFromHeader(const String& headerValue, const URL& baseURL, Document&, MediaAttributeCheck);
    static bool isSupportedType(CachedResource::Type, const String& mimeType, Document&);

    void triggerEvents(const CachedResource&);
    void triggerError();
    void cancelLoad();

private:
    void notifyFinished(CachedResource&, const NetworkLoadMetrics&, LoadWillContinueInAnotherProcess) override;
    static void preconnectIfNeeded(const LinkLoadParameters&, Document&);
    static std::unique_ptr<LinkPreloadResourceClient> preloadIfNeeded(const LinkLoadParameters&, Document&, LinkLoader*);
    void prefetchIfNeeded(const LinkLoadParameters&, Document&);

    WeakRef<LinkLoaderClient> m_client;
    CachedResourceHandle<CachedResource> m_cachedLinkResource;
    std::unique_ptr<LinkPreloadResourceClient> m_preloadResourceClient;
};

}
