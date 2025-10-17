/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 21, 2022.
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

#include "PublicSuffixStore.h"
#include "SecurityOriginData.h"
#include <wtf/HashTraits.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/URL.h>
#include <wtf/text/StringHash.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class RegistrableDomain {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(RegistrableDomain);
public:
    RegistrableDomain() = default;

    explicit RegistrableDomain(const URL& url)
        : RegistrableDomain(registrableDomainFromHost(url.host().toString()))
    {
    }

    explicit RegistrableDomain(const SecurityOriginData& origin)
        : RegistrableDomain(registrableDomainFromHost(origin.host()))
    {
    }

    static RegistrableDomain fromRawString(String&& origin)
    {
        return RegistrableDomain(WTFMove(origin));
    }

    bool isEmpty() const { return m_registrableDomain.isEmpty() || m_registrableDomain == "nullOrigin"_s; }
    const String& string() const { return m_registrableDomain; }

    friend bool operator==(const RegistrableDomain&, const RegistrableDomain&) = default;
    bool operator==(ASCIILiteral other) const { return m_registrableDomain == other; }

    bool matches(const URL& url) const
    {
        return matches(url.host());
    }

    bool matches(const SecurityOriginData& origin) const
    {
        return matches(origin.host());
    }

    RegistrableDomain isolatedCopy() const & { return RegistrableDomain { m_registrableDomain.isolatedCopy() }; }
    RegistrableDomain isolatedCopy() && { return RegistrableDomain { WTFMove(m_registrableDomain).isolatedCopy() }; }

    RegistrableDomain(WTF::HashTableDeletedValueType)
        : m_registrableDomain(WTF::HashTableDeletedValue) { }
    bool isHashTableDeletedValue() const { return m_registrableDomain.isHashTableDeletedValue(); }
    unsigned hash() const { return m_registrableDomain.hash(); }

    struct RegistrableDomainHash {
        static unsigned hash(const RegistrableDomain& registrableDomain) { return ASCIICaseInsensitiveHash::hash(registrableDomain.m_registrableDomain.impl()); }
        static bool equal(const RegistrableDomain& a, const RegistrableDomain& b) { return equalIgnoringASCIICase(a.string(), b.string()); }
        static const bool safeToCompareToEmptyOrDeleted = false;
    };

    static RegistrableDomain uncheckedCreateFromRegistrableDomainString(const String& domain)
    {
        return RegistrableDomain { String { domain } };
    }
    
    static RegistrableDomain uncheckedCreateFromHost(const String& host)
    {
        auto registrableDomain = PublicSuffixStore::singleton().topPrivatelyControlledDomain(host);
        if (registrableDomain.isEmpty())
            return uncheckedCreateFromRegistrableDomainString(host);
        return RegistrableDomain { WTFMove(registrableDomain) };
    }

private:
    explicit RegistrableDomain(String&& domain)
        : m_registrableDomain { domain.isEmpty() ? "nullOrigin"_s : WTFMove(domain) }
    {
    }

    bool matches(StringView host) const
    {
        if (host.isEmpty() && m_registrableDomain == "nullOrigin"_s)
            return true;
        if (!host.endsWith(m_registrableDomain))
            return false;
        if (host.length() == m_registrableDomain.length())
            return true;
        return host[host.length() - m_registrableDomain.length() - 1] == '.';
    }

    static inline String registrableDomainFromHost(const String& host)
    {
        auto domain = PublicSuffixStore::singleton().topPrivatelyControlledDomain(host);
        if (host.isEmpty())
            domain = "nullOrigin"_s;
        else if (domain.isEmpty())
            domain = host;
        return domain;
    }

    String m_registrableDomain;
};

inline bool areRegistrableDomainsEqual(const URL& a, const URL& b)
{
    return RegistrableDomain(a).matches(b);
}

} // namespace WebCore

namespace WTF {
template<> struct DefaultHash<WebCore::RegistrableDomain> : WebCore::RegistrableDomain::RegistrableDomainHash { };
template<> struct HashTraits<WebCore::RegistrableDomain> : SimpleClassHashTraits<WebCore::RegistrableDomain> { };

template<> class StringTypeAdapter<WebCore::RegistrableDomain, void> : public StringTypeAdapter<String, void> {
public:
    StringTypeAdapter(const WebCore::RegistrableDomain& domain)
        : StringTypeAdapter<String, void>(domain.string())
    { }
};

} // namespace WTF
