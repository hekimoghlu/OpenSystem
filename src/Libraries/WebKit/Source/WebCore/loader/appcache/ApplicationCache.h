/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 6, 2024.
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

#include <wtf/HashMap.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/StringHash.h>

namespace WebCore {

class ApplicationCacheGroup;
class ApplicationCacheResource;
class ResourceRequest;

using FallbackURLVector = Vector<std::pair<URL, URL>>;

class ApplicationCache : public RefCounted<ApplicationCache> {
public:
    static Ref<ApplicationCache> create() { return adoptRef(*new ApplicationCache); }

    ~ApplicationCache();

    void addResource(Ref<ApplicationCacheResource>&&);

    void setManifestResource(Ref<ApplicationCacheResource>&&);
    ApplicationCacheResource* manifestResource() const;

    void setGroup(ApplicationCacheGroup*);
    ApplicationCacheGroup* group() const;

    bool isComplete();

    ApplicationCacheResource* resourceForRequest(const ResourceRequest&);
    ApplicationCacheResource* resourceForURL(const String& url);

    void setAllowsAllNetworkRequests(bool value) { m_allowAllNetworkRequests = value; }
    bool allowsAllNetworkRequests() const { return m_allowAllNetworkRequests; }
    void setOnlineAllowlist(const Vector<URL>&);
    const Vector<URL>& onlineAllowlist() const { return m_onlineAllowlist; }
    bool isURLInOnlineAllowlist(const URL&); // There is an entry in online allowlist that has the same origin as the resource's URL and that is a prefix match for the resource's URL.

    void setFallbackURLs(const FallbackURLVector&);
    const FallbackURLVector& fallbackURLs() const { return m_fallbackURLs; }
    bool urlMatchesFallbackNamespace(const URL&, URL* fallbackURL = nullptr);

#ifndef NDEBUG
    void dump();
#endif

    using ResourceMap = HashMap<String, RefPtr<ApplicationCacheResource>>;
    const ResourceMap& resources() const { return m_resources; }

    void setStorageID(unsigned storageID) { m_storageID = storageID; }
    unsigned storageID() const { return m_storageID; }
    void clearStorageID();

    static bool requestIsHTTPOrHTTPSGet(const ResourceRequest&);

    int64_t estimatedSizeInStorage() const { return m_estimatedSizeInStorage; }

private:
    ApplicationCache();

    WeakPtr<ApplicationCacheGroup> m_group;
    ResourceMap m_resources;
    WeakPtr<ApplicationCacheResource> m_manifest;

    bool m_allowAllNetworkRequests { false };
    Vector<URL> m_onlineAllowlist;
    FallbackURLVector m_fallbackURLs;

    // The total size of the resources belonging to this Application Cache instance.
    // This is an estimation of the size this Application Cache occupies in the database file.
    int64_t m_estimatedSizeInStorage { 0 };

    unsigned m_storageID { 0 };
};

} // namespace WebCore
