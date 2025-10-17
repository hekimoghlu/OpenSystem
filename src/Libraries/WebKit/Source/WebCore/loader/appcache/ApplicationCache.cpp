/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 11, 2023.
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
#include "ApplicationCache.h"

#include "ApplicationCacheGroup.h"
#include "ApplicationCacheResource.h"
#include "ApplicationCacheStorage.h"
#include "ResourceRequest.h"
#include <algorithm>
#include <stdio.h>
#include <wtf/text/CString.h>

namespace WebCore {
 
static inline bool fallbackURLLongerThan(const std::pair<URL, URL>& lhs, const std::pair<URL, URL>& rhs)
{
    return lhs.first.string().length() > rhs.first.string().length();
}

ApplicationCache::ApplicationCache()
{
}

ApplicationCache::~ApplicationCache()
{
    if (m_group)
        m_group->cacheDestroyed(*this);
}
    
void ApplicationCache::setGroup(ApplicationCacheGroup* group)
{
    ASSERT(!m_group || group == m_group);
    m_group = group;
}

bool ApplicationCache::isComplete()
{
    return m_group && m_group->cacheIsComplete(*this);
}

ApplicationCacheResource* ApplicationCache::manifestResource() const
{
    return m_manifest.get();
}

ApplicationCacheGroup* ApplicationCache::group() const
{
    return m_group.get();
}

void ApplicationCache::setManifestResource(Ref<ApplicationCacheResource>&& manifest)
{
    ASSERT(!m_manifest);
    ASSERT(manifest->type() & ApplicationCacheResource::Manifest);

    m_manifest = manifest;

    addResource(WTFMove(manifest));
}
    
void ApplicationCache::addResource(Ref<ApplicationCacheResource>&& resource)
{
    auto& url = resource->url();

    ASSERT(!url.hasFragmentIdentifier());
    ASSERT(!m_resources.contains(url.string()));

    if (m_storageID) {
        ASSERT(!resource->storageID());
        ASSERT(resource->type() & ApplicationCacheResource::Master);

        // Add the resource to the storage.
        m_group->storage().store(resource.ptr(), this);
    }

    m_estimatedSizeInStorage += resource->estimatedSizeInStorage();

    m_resources.set(url.string(), WTFMove(resource));
}

ApplicationCacheResource* ApplicationCache::resourceForURL(const String& url)
{
    ASSERT(!URL({ }, url).hasFragmentIdentifier());
    return m_resources.get(url);
}    

bool ApplicationCache::requestIsHTTPOrHTTPSGet(const ResourceRequest& request)
{
    return request.url().protocolIsInHTTPFamily() && equalLettersIgnoringASCIICase(request.httpMethod(), "get"_s);
}

ApplicationCacheResource* ApplicationCache::resourceForRequest(const ResourceRequest& request)
{
    // We only care about HTTP/HTTPS GET requests.
    if (!requestIsHTTPOrHTTPSGet(request))
        return nullptr;

    URL url(request.url());
    url.removeFragmentIdentifier();
    return resourceForURL(url.string());
}

void ApplicationCache::setOnlineAllowlist(const Vector<URL>& onlineAllowlist)
{
    ASSERT(m_onlineAllowlist.isEmpty());
    m_onlineAllowlist = onlineAllowlist;
}

bool ApplicationCache::isURLInOnlineAllowlist(const URL& url)
{
    for (auto& allowlistURL : m_onlineAllowlist) {
        if (protocolHostAndPortAreEqual(url, allowlistURL) && url.string().startsWith(allowlistURL.string()))
            return true;
    }
    return false;
}

void ApplicationCache::setFallbackURLs(const FallbackURLVector& fallbackURLs)
{
    ASSERT(m_fallbackURLs.isEmpty());
    m_fallbackURLs = fallbackURLs;
    // FIXME: What's the right behavior if we have 2 or more identical namespace URLs?
    std::stable_sort(m_fallbackURLs.begin(), m_fallbackURLs.end(), fallbackURLLongerThan);
}

bool ApplicationCache::urlMatchesFallbackNamespace(const URL& url, URL* fallbackURL)
{
    for (auto& fallback : m_fallbackURLs) {
        if (protocolHostAndPortAreEqual(url, fallback.first) && url.string().startsWith(fallback.first.string())) {
            if (fallbackURL)
                *fallbackURL = fallback.second;
            return true;
        }
    }
    return false;
}

void ApplicationCache::clearStorageID()
{
    m_storageID = 0;
    
    for (const auto& resource : m_resources.values())
        resource->clearStorageID();
}
    
#ifndef NDEBUG
void ApplicationCache::dump()
{
    for (const auto& urlAndResource : m_resources) {
        SAFE_PRINTF("%s ", urlAndResource.key.utf8());
        ApplicationCacheResource::dumpType(urlAndResource.value->type());
    }
}
#endif

}
