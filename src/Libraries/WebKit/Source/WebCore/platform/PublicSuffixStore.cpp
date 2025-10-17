/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 21, 2023.
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
#include "PublicSuffixStore.h"

#include <wtf/CrossThreadCopier.h>
#include <wtf/NeverDestroyed.h>

namespace WebCore {

PublicSuffixStore& PublicSuffixStore::singleton()
{
    static LazyNeverDestroyed<PublicSuffixStore> store;
    static std::once_flag flag;
    std::call_once(flag, [&] {
        store.construct();
    });
    return store.get();
}

void PublicSuffixStore::clearHostTopPrivatelyControlledDomainCache()
{
    Locker locker { m_HostTopPrivatelyControlledDomainCacheLock };
    m_hostTopPrivatelyControlledDomainCache.clear();
}

bool PublicSuffixStore::isPublicSuffix(StringView domain) const
{
    return platformIsPublicSuffix(domain);
}

PublicSuffix PublicSuffixStore::publicSuffix(const URL& url) const
{
    if (!url.isValid())
        return { };

    auto host = url.host();
    if (URL::hostIsIPAddress(host))
        return { };

    size_t separatorPosition;
    for (unsigned labelStart = 0; (separatorPosition = host.find('.', labelStart)) != notFound; labelStart = separatorPosition + 1) {
        auto candidate = host.substring(separatorPosition + 1);
        if (isPublicSuffix(candidate))
            return PublicSuffix::fromRawString(candidate.toString());
    }

    return { };
}

String PublicSuffixStore::topPrivatelyControlledDomain(StringView host) const
{
    // FIXME: if host is a URL, we could drop these checks.
    if (host.isEmpty())
        return { };

    if (!host.containsOnlyASCII())
        return host.toString();

    Locker locker { m_HostTopPrivatelyControlledDomainCacheLock };
    auto result = m_hostTopPrivatelyControlledDomainCache.ensure<ASCIICaseInsensitiveStringViewHashTranslator>(host, [&] {
        const auto lowercaseHost = host.convertToASCIILowercase();
        if (lowercaseHost == "localhost"_s || URL::hostIsIPAddress(lowercaseHost))
            return lowercaseHost;

        return platformTopPrivatelyControlledDomain(lowercaseHost);
    }).iterator->value.isolatedCopy();

    constexpr auto maxHostTopPrivatelyControlledDomainCache = 128;
    if (m_hostTopPrivatelyControlledDomainCache.size() > maxHostTopPrivatelyControlledDomainCache)
        m_hostTopPrivatelyControlledDomainCache.remove(m_hostTopPrivatelyControlledDomainCache.random());

    return result;
}

String PublicSuffixStore::topPrivatelyControlledDomainWithoutPublicSuffix(StringView host) const
{
    auto topPrivatelyControlledDomain = this->topPrivatelyControlledDomain(host);
    if (topPrivatelyControlledDomain.isEmpty())
        return { };

    return domainWithoutPublicSuffix(topPrivatelyControlledDomain);
}

String PublicSuffixStore::domainWithoutPublicSuffix(StringView domain) const
{
    if (URL::hostIsIPAddress(domain))
        return domain.toString();

    size_t separatorPosition;
    for (unsigned labelStart = 0; (separatorPosition = domain.find('.', labelStart)) != notFound; labelStart = separatorPosition + 1) {
        auto candidate = domain.substring(separatorPosition + 1);
        if (isPublicSuffix(candidate)) {
            if (separatorPosition > 0 && domain.length() > separatorPosition) {
                auto domainWithoutSuffix = domain.substring(0, separatorPosition);
                return domainWithoutSuffix.toString();
            }
        }
    }

    return domain.toString();
}

} // namespace WebCore
