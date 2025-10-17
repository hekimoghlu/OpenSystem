/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 2, 2023.
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

#include "ClientOrigin.h"
#include "LoaderMalloc.h"
#include "StoredCredentialsPolicy.h"
#include <pal/SessionID.h>
#include <wtf/Expected.h>
#include <wtf/HashMap.h>
#include <wtf/HashSet.h>
#include <wtf/MonotonicTime.h>
#include <wtf/URLHash.h>
#include <wtf/UniqueRef.h>

namespace WebCore {

class HTTPHeaderMap;
class ResourceResponse;

class CrossOriginPreflightResultCacheItem {
    WTF_MAKE_NONCOPYABLE(CrossOriginPreflightResultCacheItem); WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(Loader);
public:
    static Expected<UniqueRef<CrossOriginPreflightResultCacheItem>, String> create(StoredCredentialsPolicy, const ResourceResponse&);

    CrossOriginPreflightResultCacheItem(MonotonicTime, StoredCredentialsPolicy, UncheckedKeyHashSet<String>&&, UncheckedKeyHashSet<String, ASCIICaseInsensitiveHash>&&);

    std::optional<String> validateMethodAndHeaders(const String& method, const HTTPHeaderMap&) const;
    bool allowsRequest(StoredCredentialsPolicy, const String& method, const HTTPHeaderMap&) const;

private:
    bool allowsCrossOriginMethod(const String&, StoredCredentialsPolicy) const;
    std::optional<String> validateCrossOriginHeaders(const HTTPHeaderMap&, StoredCredentialsPolicy) const;

    // FIXME: A better solution to holding onto the absolute expiration time might be
    // to start a timer for the expiration delta that removes this from the cache when
    // it fires.
    MonotonicTime m_absoluteExpiryTime;
    StoredCredentialsPolicy m_storedCredentialsPolicy;
    UncheckedKeyHashSet<String> m_methods;
    UncheckedKeyHashSet<String, ASCIICaseInsensitiveHash> m_headers;
};

class CrossOriginPreflightResultCache {
    WTF_MAKE_NONCOPYABLE(CrossOriginPreflightResultCache); WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(Loader);
public:
    WEBCORE_EXPORT static CrossOriginPreflightResultCache& singleton();
    WEBCORE_EXPORT void appendEntry(PAL::SessionID, const ClientOrigin&, const URL&, std::unique_ptr<CrossOriginPreflightResultCacheItem>);
    WEBCORE_EXPORT bool canSkipPreflight(PAL::SessionID, const ClientOrigin&, const URL&, StoredCredentialsPolicy, const String& method, const HTTPHeaderMap& requestHeaders);
    WEBCORE_EXPORT void clear();

private:
    friend NeverDestroyed<CrossOriginPreflightResultCache>;
    CrossOriginPreflightResultCache();

    HashMap<std::tuple<PAL::SessionID, ClientOrigin, URL>, std::unique_ptr<CrossOriginPreflightResultCacheItem>> m_preflightHashMap;
};

inline CrossOriginPreflightResultCacheItem::CrossOriginPreflightResultCacheItem(MonotonicTime absoluteExpiryTime, StoredCredentialsPolicy  storedCredentialsPolicy, UncheckedKeyHashSet<String>&& methods, UncheckedKeyHashSet<String, ASCIICaseInsensitiveHash>&& headers)
    : m_absoluteExpiryTime(absoluteExpiryTime)
    , m_storedCredentialsPolicy(storedCredentialsPolicy)
    , m_methods(WTFMove(methods))
    , m_headers(WTFMove(headers))
{
}

} // namespace WebCore
