/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 23, 2023.
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
#import "config.h"
#import "PublicSuffixStore.h"

#import <pal/spi/cf/CFNetworkSPI.h>
#import <wtf/cocoa/NSURLExtras.h>

namespace WebCore {

static bool isPublicSuffixCF(const String& domain)
{
    NSString *host = WTF::decodeHostName(domain);
    return host && _CFHostIsDomainTopLevel((__bridge CFStringRef)host);
}

bool PublicSuffixStore::platformIsPublicSuffix(StringView domain) const
{
    auto domainString = domain.toStringWithoutCopying();
    {
        Locker locker { m_publicSuffixCacheLock };
        if (m_publicSuffixCache) {
            auto publicSuffix = PublicSuffix::fromRawString(String { domainString });
            if (m_publicSuffixCache->contains(publicSuffix))
                return true;
        }
    }

    return isPublicSuffixCF(domainString);
}

String PublicSuffixStore::platformTopPrivatelyControlledDomain(StringView host) const
{
    @autoreleasepool {
        size_t separatorPosition;
        for (unsigned labelStart = 0; (separatorPosition = host.find('.', labelStart)) != notFound; labelStart = separatorPosition + 1) {
            if (isPublicSuffix(host.substring(separatorPosition + 1)))
                return host.substring(labelStart).toString();
        }

        return { };
    }
}

void PublicSuffixStore::enablePublicSuffixCache()
{
    RELEASE_ASSERT(isMainThread());

    Locker locker { m_publicSuffixCacheLock };
    ASSERT(!m_publicSuffixCache);
    m_publicSuffixCache = UncheckedKeyHashSet<PublicSuffix> { };
}

void PublicSuffixStore::addPublicSuffix(const PublicSuffix& publicSuffix)
{
    RELEASE_ASSERT(isMainThread());

    if (!publicSuffix.isValid())
        return;

    Locker locker { m_publicSuffixCacheLock };
    ASSERT(m_publicSuffixCache);
    m_publicSuffixCache->add(crossThreadCopy(publicSuffix));
}

} // namespace WebCore
