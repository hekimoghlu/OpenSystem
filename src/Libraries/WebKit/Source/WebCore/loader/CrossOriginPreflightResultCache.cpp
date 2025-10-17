/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 20, 2025.
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
#include "CrossOriginPreflightResultCache.h"

#include "CrossOriginAccessControl.h"
#include "HTTPHeaderNames.h"
#include "HTTPParsers.h"
#include "ResourceResponse.h"
#include <wtf/MainThread.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/text/MakeString.h>
#include <wtf/text/StringToIntegerConversion.h>

namespace WebCore {

// These values are at the discretion of the user agent.
static const auto defaultPreflightCacheTimeout = 5_s;
static const auto maxPreflightCacheTimeout = 600_s; // Should be short enough to minimize the risk of using a poisoned cache after switching to a secure network.

CrossOriginPreflightResultCache::CrossOriginPreflightResultCache()
{
}

static bool parseAccessControlMaxAge(const String& string, Seconds& expiryDelta)
{
    // FIXME: This should probably reject strings that have a leading "+".
    auto parsedInteger = parseInteger<int64_t>(string);
    expiryDelta = Seconds(static_cast<double>(parsedInteger.value_or(0)));
    return parsedInteger.has_value();
}

Expected<UniqueRef<CrossOriginPreflightResultCacheItem>, String> CrossOriginPreflightResultCacheItem::create(StoredCredentialsPolicy policy, const ResourceResponse& response)
{
    auto methods = parseAccessControlAllowList(response.httpHeaderField(HTTPHeaderName::AccessControlAllowMethods));
    if (!methods)
        return makeUnexpected(makeString("Header Access-Control-Allow-Methods has an invalid value: "_s, response.httpHeaderField(HTTPHeaderName::AccessControlAllowMethods)));

    auto headers = parseAccessControlAllowList<ASCIICaseInsensitiveHash>(response.httpHeaderField(HTTPHeaderName::AccessControlAllowHeaders));
    if (!headers)
        return makeUnexpected(makeString("Header Access-Control-Allow-Headers has an invalid value: "_s, response.httpHeaderField(HTTPHeaderName::AccessControlAllowHeaders)));

    Seconds expiryDelta = 0_s;
    if (parseAccessControlMaxAge(response.httpHeaderField(HTTPHeaderName::AccessControlMaxAge), expiryDelta)) {
        if (expiryDelta > maxPreflightCacheTimeout)
            expiryDelta = maxPreflightCacheTimeout;
    } else
        expiryDelta = defaultPreflightCacheTimeout;

    return makeUniqueRef<CrossOriginPreflightResultCacheItem>(MonotonicTime::now() + expiryDelta, policy, WTFMove(*methods), WTFMove(*headers));
}

std::optional<String> CrossOriginPreflightResultCacheItem::validateMethodAndHeaders(const String& method, const HTTPHeaderMap& requestHeaders) const
{
    if (!allowsCrossOriginMethod(method, m_storedCredentialsPolicy))
        return makeString("Method "_s, method, " is not allowed by Access-Control-Allow-Methods."_s);

    if (auto badHeader = validateCrossOriginHeaders(requestHeaders, m_storedCredentialsPolicy))
        return makeString("Request header field "_s, *badHeader, " is not allowed by Access-Control-Allow-Headers."_s);
    return { };
}

bool CrossOriginPreflightResultCacheItem::allowsCrossOriginMethod(const String& method, StoredCredentialsPolicy storedCredentialsPolicy) const
{
    return m_methods.contains(method) || (m_methods.contains<HashTranslatorASCIILiteral>("*"_s) && storedCredentialsPolicy != StoredCredentialsPolicy::Use) || isOnAccessControlSimpleRequestMethodAllowlist(method);
}

std::optional<String> CrossOriginPreflightResultCacheItem::validateCrossOriginHeaders(const HTTPHeaderMap& requestHeaders, StoredCredentialsPolicy storedCredentialsPolicy) const
{
    bool validWildcard = m_headers.contains<HashTranslatorASCIILiteral>("*"_s) && storedCredentialsPolicy != StoredCredentialsPolicy::Use;
    for (const auto& header : requestHeaders) {
        if (header.keyAsHTTPHeaderName && isCrossOriginSafeRequestHeader(header.keyAsHTTPHeaderName.value(), header.value))
            continue;
        if (!m_headers.contains(header.key) && !validWildcard)
            return header.key;
    }
    return { };
}

bool CrossOriginPreflightResultCacheItem::allowsRequest(StoredCredentialsPolicy storedCredentialsPolicy, const String& method, const HTTPHeaderMap& requestHeaders) const
{
    if (m_absoluteExpiryTime < MonotonicTime::now())
        return false;
    if (storedCredentialsPolicy == StoredCredentialsPolicy::Use && m_storedCredentialsPolicy == StoredCredentialsPolicy::DoNotUse)
        return false;
    if (!allowsCrossOriginMethod(method, storedCredentialsPolicy))
        return false;
    if (auto badHeader = validateCrossOriginHeaders(requestHeaders, storedCredentialsPolicy))
        return false;
    return true;
}

CrossOriginPreflightResultCache& CrossOriginPreflightResultCache::singleton()
{
    ASSERT(isMainThread());

    static NeverDestroyed<CrossOriginPreflightResultCache> cache;
    return cache;
}

void CrossOriginPreflightResultCache::appendEntry(PAL::SessionID sessionID, const ClientOrigin& origin, const URL& url, std::unique_ptr<CrossOriginPreflightResultCacheItem> preflightResult)
{
    ASSERT(isMainThread());
    m_preflightHashMap.set(std::make_tuple(sessionID, origin, url), WTFMove(preflightResult));
}

bool CrossOriginPreflightResultCache::canSkipPreflight(PAL::SessionID sessionID, const ClientOrigin& origin, const URL& url, StoredCredentialsPolicy storedCredentialsPolicy, const String& method, const HTTPHeaderMap& requestHeaders)
{
    ASSERT(isMainThread());
    auto it = m_preflightHashMap.find(std::make_tuple(sessionID, origin, url));
    if (it == m_preflightHashMap.end())
        return false;

    if (it->value->allowsRequest(storedCredentialsPolicy, method, requestHeaders))
        return true;

    m_preflightHashMap.remove(it);
    return false;
}

void CrossOriginPreflightResultCache::clear()
{
    ASSERT(isMainThread());
    m_preflightHashMap.clear();
}

} // namespace WebCore
