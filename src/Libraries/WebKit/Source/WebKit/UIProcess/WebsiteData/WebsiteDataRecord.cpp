/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 5, 2022.
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
#include "WebsiteDataRecord.h"

#include <WebCore/LocalizedStrings.h>
#include <WebCore/PublicSuffixStore.h>
#include <WebCore/SecurityOrigin.h>
#include <wtf/CrossThreadCopier.h>

#if PLATFORM(COCOA)
#import <pal/spi/cf/CFNetworkSPI.h>
#endif

static String displayNameForLocalFiles()
{
    return WEB_UI_STRING("Local documents on your computer", "'Website' name displayed when local documents have stored local data");
}

namespace WebKit {

String WebsiteDataRecord::displayNameForCookieHostName(const String& hostName)
{
#if PLATFORM(COCOA)
    if (hostName == String(kCFHTTPCookieLocalFileDomain))
        return displayNameForLocalFiles();
#else
    if (hostName == "localhost"_s)
        return hostName;
#endif
    return displayNameForHostName(hostName);
}

String WebsiteDataRecord::displayNameForHostName(const String& hostName)
{
    return WebCore::PublicSuffixStore::singleton().topPrivatelyControlledDomain(hostName);
}

String WebsiteDataRecord::displayNameForOrigin(const WebCore::SecurityOriginData& securityOrigin)
{
    const auto& protocol = securityOrigin.protocol();

    if (protocol == "file"_s)
        return displayNameForLocalFiles();

    if (protocol == "http"_s || protocol == "https"_s)
        return WebCore::PublicSuffixStore::singleton().topPrivatelyControlledDomain(securityOrigin.host());

    return String();
}

void WebsiteDataRecord::add(WebsiteDataType type, const WebCore::SecurityOriginData& origin)
{
    types.add(type);
    origins.add(origin);
}

void WebsiteDataRecord::addCookieHostName(const String& hostName)
{
    types.add(WebsiteDataType::Cookies);
    cookieHostNames.add(hostName);
}

void WebsiteDataRecord::addHSTSCacheHostname(const String& hostName)
{
    types.add(WebsiteDataType::HSTSCache);
    HSTSCacheHostNames.add(hostName);
}

void WebsiteDataRecord::addAlternativeServicesHostname(const String& hostName)
{
#if HAVE(ALTERNATIVE_SERVICE)
    types.add(WebsiteDataType::AlternativeServices);
    alternativeServicesHostNames.add(hostName);
#else
    UNUSED_PARAM(hostName);
#endif
}

void WebsiteDataRecord::addResourceLoadStatisticsRegistrableDomain(const WebCore::RegistrableDomain& domain)
{
    types.add(WebsiteDataType::ResourceLoadStatistics);
    resourceLoadStatisticsRegistrableDomains.add(domain);
}

static inline bool hostIsInDomain(StringView host, StringView domain)
{
    if (!host.endsWithIgnoringASCIICase(domain))
        return false;
    
    ASSERT(host.length() >= domain.length());
    unsigned suffixOffset = host.length() - domain.length();
    return !suffixOffset || host[suffixOffset - 1] == '.';
}

bool WebsiteDataRecord::matches(const WebCore::RegistrableDomain& domain) const
{
    if (domain.isEmpty())
        return false;

    if (types.contains(WebsiteDataType::Cookies)) {
        for (const auto& hostName : cookieHostNames) {
            if (hostIsInDomain(hostName, domain.string()))
                return true;
        }
    }

    for (const auto& dataRecordOriginData : origins) {
        if (hostIsInDomain(dataRecordOriginData.host(), domain.string()))
            return true;
    }

    return false;
}

String WebsiteDataRecord::topPrivatelyControlledDomain()
{
    auto& publicSuffixStore = WebCore::PublicSuffixStore::singleton();
    if (!cookieHostNames.isEmpty())
        return publicSuffixStore.topPrivatelyControlledDomain(cookieHostNames.takeAny());
    
    if (!origins.isEmpty())
        return publicSuffixStore.topPrivatelyControlledDomain(origins.takeAny().securityOrigin().get().host());
    
    return emptyString();
}

WebsiteDataRecord WebsiteDataRecord::isolatedCopy() const &
{
    return WebsiteDataRecord {
        crossThreadCopy(displayName),
        types,
        size,
        crossThreadCopy(origins),
        crossThreadCopy(cookieHostNames),
        crossThreadCopy(HSTSCacheHostNames),
        crossThreadCopy(alternativeServicesHostNames),
        crossThreadCopy(resourceLoadStatisticsRegistrableDomains),
    };
}

WebsiteDataRecord WebsiteDataRecord::isolatedCopy() &&
{
    return WebsiteDataRecord {
        crossThreadCopy(WTFMove(displayName)),
        types,
        size,
        crossThreadCopy(WTFMove(origins)),
        crossThreadCopy(WTFMove(cookieHostNames)),
        crossThreadCopy(WTFMove(HSTSCacheHostNames)),
        crossThreadCopy(WTFMove(alternativeServicesHostNames)),
        crossThreadCopy(WTFMove(resourceLoadStatisticsRegistrableDomains)),
    };
}

}
