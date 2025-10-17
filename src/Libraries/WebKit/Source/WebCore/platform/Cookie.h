/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 30, 2022.
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

#include <wtf/URL.h>
#include <wtf/text/StringHash.h>
#include <wtf/text/WTFString.h>

#ifdef __OBJC__
#include <objc/objc.h>
#endif

#if USE(SOUP)
typedef struct _SoupCookie SoupCookie;
#endif

namespace WebCore {

struct Cookie {
    Cookie() = default;

    WEBCORE_EXPORT bool operator==(const Cookie&) const;
    WEBCORE_EXPORT unsigned hash() const;

#ifdef __OBJC__
    WEBCORE_EXPORT Cookie(NSHTTPCookie *);
    WEBCORE_EXPORT operator NSHTTPCookie *() const;
#elif USE(SOUP)
    explicit Cookie(SoupCookie*);
    SoupCookie* toSoupCookie() const;
#endif

    bool isNull() const
    {
        return name.isNull()
            && value.isNull()
            && domain.isNull()
            && path.isNull()
            && partitionKey.isNull()
            && !created
            && !expires
            && !httpOnly
            && !secure
            && !session
            && comment.isNull()
            && commentURL.isNull();
    }
    
    bool isKeyEqual(const Cookie& otherCookie) const
    {
        return name == otherCookie.name
            && domain == otherCookie.domain
            && path == otherCookie.path;
    }

    String name;
    String value;
    String domain;
    String path;
    String partitionKey;
    // Creation and expiration dates are expressed as milliseconds since the UNIX epoch.
    double created { 0 };
    std::optional<double> expires;
    bool httpOnly { false };
    bool secure { false };
    bool session { false };
    String comment;
    URL commentURL;
    Vector<uint16_t> ports;

    enum class SameSitePolicy : uint8_t { 
        None, 
        Lax, 
        Strict 
    };

    SameSitePolicy sameSite { SameSitePolicy::None };

    Cookie(String&& name, String&& value, String&& domain, String&& path, String&& partitionKey, double created, std::optional<double> expires, bool httpOnly, bool secure, bool session, String&& comment, URL&& commentURL, Vector<uint16_t> ports, SameSitePolicy sameSite)
        : name(WTFMove(name))
        , value(WTFMove(value))
        , domain(WTFMove(domain))
        , path(WTFMove(path))
        , partitionKey(WTFMove(partitionKey))
        , created(created)
        , expires(expires)
        , httpOnly(httpOnly)
        , secure(secure)
        , session(session)
        , comment(WTFMove(comment))
        , commentURL(WTFMove(commentURL))
        , ports(WTFMove(ports))
        , sameSite(sameSite)
    {
    }

    Cookie isolatedCopy() const & { return { name.isolatedCopy(), value.isolatedCopy(), domain.isolatedCopy(), path.isolatedCopy(), partitionKey.isolatedCopy(), created, expires, httpOnly, secure, session, comment.isolatedCopy(), commentURL.isolatedCopy(), ports, sameSite }; }
    Cookie isolatedCopy() && { return { WTFMove(name).isolatedCopy(), WTFMove(value).isolatedCopy(), WTFMove(domain).isolatedCopy(), WTFMove(path).isolatedCopy(), WTFMove(partitionKey).isolatedCopy(), created, expires, httpOnly, secure, session, WTFMove(comment).isolatedCopy(), WTFMove(commentURL).isolatedCopy(), WTFMove(ports), sameSite }; }
};

struct CookieHash {
    static unsigned hash(const Cookie& key)
    {
        return key.hash();
    }

    static bool equal(const Cookie& a, const Cookie& b)
    {
        return a == b;
    }
    static const bool safeToCompareToEmptyOrDeleted = false;
};

}

namespace WTF {
    template<typename T> struct DefaultHash;
    template<> struct DefaultHash<WebCore::Cookie> : WebCore::CookieHash { };
    template<> struct HashTraits<WebCore::Cookie> : GenericHashTraits<WebCore::Cookie> {
        static WebCore::Cookie emptyValue() { return { }; }
        static void constructDeletedValue(WebCore::Cookie& slot) { new (NotNull, &slot.name) String(WTF::HashTableDeletedValue); }
        static bool isDeletedValue(const WebCore::Cookie& slot) { return slot.name.isHashTableDeletedValue(); }

        static const bool hasIsEmptyValueFunction = true;
        static bool isEmptyValue(const WebCore::Cookie& slot) { return slot.isNull(); }
    };
}
