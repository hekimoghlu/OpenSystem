/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 8, 2025.
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
#include "WebsiteData.h"

#include "ArgumentCoders.h"
#include "WebsiteDataType.h"
#include <WebCore/RegistrableDomain.h>
#include <WebCore/SecurityOriginData.h>
#include <wtf/CrossThreadCopier.h>
#include <wtf/text/StringHash.h>

namespace WebKit {

WebsiteDataProcessType WebsiteData::ownerProcess(WebsiteDataType dataType)
{
    switch (dataType) {
    case WebsiteDataType::Cookies:
        return WebsiteDataProcessType::Network;
    case WebsiteDataType::DiskCache:
        return WebsiteDataProcessType::Network;
    case WebsiteDataType::MemoryCache:
        return WebsiteDataProcessType::Web;
    case WebsiteDataType::OfflineWebApplicationCache:
        return WebsiteDataProcessType::UI;
    case WebsiteDataType::SessionStorage:
        return WebsiteDataProcessType::Network;
    case WebsiteDataType::LocalStorage:
        return WebsiteDataProcessType::Network;
    case WebsiteDataType::WebSQLDatabases:
        return WebsiteDataProcessType::UI;
    case WebsiteDataType::IndexedDBDatabases:
        return WebsiteDataProcessType::Network;
    case WebsiteDataType::MediaKeys:
        return WebsiteDataProcessType::UI;
    case WebsiteDataType::HSTSCache:
        return WebsiteDataProcessType::Network;
    case WebsiteDataType::SearchFieldRecentSearches:
        return WebsiteDataProcessType::UI;
    case WebsiteDataType::ResourceLoadStatistics:
        return WebsiteDataProcessType::Network;
    case WebsiteDataType::Credentials:
        return WebsiteDataProcessType::Network;
    case WebsiteDataType::ServiceWorkerRegistrations:
    case WebsiteDataType::BackgroundFetchStorage:
        return WebsiteDataProcessType::Network;
    case WebsiteDataType::DOMCache:
        return WebsiteDataProcessType::Network;
    case WebsiteDataType::DeviceIdHashSalt:
        return WebsiteDataProcessType::UI;
    case WebsiteDataType::PrivateClickMeasurements:
        return WebsiteDataProcessType::Network;
#if HAVE(ALTERNATIVE_SERVICE)
    case WebsiteDataType::AlternativeServices:
        return WebsiteDataProcessType::Network;
#endif
    case WebsiteDataType::FileSystem:
        return WebsiteDataProcessType::Network;
    }

    RELEASE_ASSERT_NOT_REACHED();
}

OptionSet<WebsiteDataType> WebsiteData::filter(OptionSet<WebsiteDataType> unfilteredWebsiteDataTypes, WebsiteDataProcessType WebsiteDataProcessType)
{
    OptionSet<WebsiteDataType> filtered;
    for (auto dataType : unfilteredWebsiteDataTypes) {
        if (ownerProcess(dataType) == WebsiteDataProcessType)
            filtered.add(dataType);
    }
    
    return filtered;
}

WebsiteData WebsiteData::isolatedCopy() const &
{
    return WebsiteData {
        crossThreadCopy(entries),
        crossThreadCopy(hostNamesWithCookies),
        crossThreadCopy(hostNamesWithHSTSCache),
        crossThreadCopy(registrableDomainsWithResourceLoadStatistics),
    };
}

WebsiteData WebsiteData::isolatedCopy() &&
{
    return WebsiteData {
        crossThreadCopy(WTFMove(entries)),
        crossThreadCopy(WTFMove(hostNamesWithCookies)),
        crossThreadCopy(WTFMove(hostNamesWithHSTSCache)),
        crossThreadCopy(WTFMove(registrableDomainsWithResourceLoadStatistics)),
    };
}

WebsiteData::Entry::Entry(WebCore::SecurityOriginData inOrigin, WebsiteDataType inType, uint64_t inSize)
    : origin(inOrigin)
    , type(inType)
    , size(inSize)
{
}

WebsiteData::Entry::Entry(WebCore::SecurityOriginData&& inOrigin, OptionSet<WebsiteDataType>&& inType, uint64_t inSize)
    : origin(WTFMove(inOrigin))
    , size(inSize)
{
    RELEASE_ASSERT(inType.hasExactlyOneBitSet());
    type = *inType.toSingleValue();
}

auto WebsiteData::Entry::isolatedCopy() const & -> Entry
{
    return { crossThreadCopy(origin), crossThreadCopy(type), size };
}

auto WebsiteData::Entry::isolatedCopy() && -> Entry
{
    return { crossThreadCopy(WTFMove(origin)), crossThreadCopy(WTFMove(type)), size };
}

}
