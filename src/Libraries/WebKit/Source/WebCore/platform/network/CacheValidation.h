/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 10, 2022.
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

#include <wtf/Forward.h>
#include <wtf/Markable.h>
#include <wtf/WallTime.h>

namespace WebCore {

class CookieJar;
class HTTPHeaderMap;
class NetworkStorageSession;
class ResourceRequest;
class ResourceResponse;

struct RedirectChainCacheStatus {
    enum class Status : uint8_t {
        NoRedirection,
        NotCachedRedirection,
        CachedRedirection
    };
    RedirectChainCacheStatus()
        : endOfValidity(WallTime::infinity())
        , status(Status::NoRedirection)
    { }
    WallTime endOfValidity;
    Status status;
};

WEBCORE_EXPORT Seconds computeCurrentAge(const ResourceResponse&, WallTime responseTimestamp);
WEBCORE_EXPORT Seconds computeFreshnessLifetimeForHTTPFamily(const ResourceResponse&, WallTime responseTimestamp);
WEBCORE_EXPORT void updateResponseHeadersAfterRevalidation(ResourceResponse&, const ResourceResponse& validatingResponse);
WEBCORE_EXPORT void updateRedirectChainStatus(RedirectChainCacheStatus&, const ResourceResponse&);

enum ReuseExpiredRedirectionOrNot { DoNotReuseExpiredRedirection, ReuseExpiredRedirection };
WEBCORE_EXPORT bool redirectChainAllowsReuse(RedirectChainCacheStatus, ReuseExpiredRedirectionOrNot);

struct CacheControlDirectives {
    constexpr CacheControlDirectives()
        : noCache(false)
        , noStore(false)
        , mustRevalidate(false)
        , immutable(false)
        { }

    Markable<Seconds, Seconds::MarkableTraits> maxAge;
    Markable<Seconds, Seconds::MarkableTraits> maxStale;
    Markable<Seconds, Seconds::MarkableTraits> staleWhileRevalidate;
    bool noCache : 1;
    bool noStore : 1;
    bool mustRevalidate : 1;
    bool immutable : 1;
};
WEBCORE_EXPORT CacheControlDirectives parseCacheControlDirectives(const HTTPHeaderMap&);

WEBCORE_EXPORT Vector<std::pair<String, String>> collectVaryingRequestHeaders(NetworkStorageSession*, const ResourceRequest&, const ResourceResponse&);
WEBCORE_EXPORT Vector<std::pair<String, String>> collectVaryingRequestHeaders(const CookieJar*, const ResourceRequest&, const ResourceResponse&);
WEBCORE_EXPORT bool verifyVaryingRequestHeaders(NetworkStorageSession*, const Vector<std::pair<String, String>>& varyingRequestHeaders, const ResourceRequest&);
WEBCORE_EXPORT bool verifyVaryingRequestHeaders(const CookieJar*, const Vector<std::pair<String, String>>& varyingRequestHeaders, const ResourceRequest&);

WEBCORE_EXPORT bool isStatusCodeCacheableByDefault(int statusCode);
WEBCORE_EXPORT bool isStatusCodePotentiallyCacheable(int statusCode);

}
