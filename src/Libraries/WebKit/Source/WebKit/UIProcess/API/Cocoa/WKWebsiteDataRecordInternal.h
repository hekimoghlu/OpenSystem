/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 1, 2024.
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
#import "WKWebsiteDataRecordPrivate.h"

#import "APIWebsiteDataRecord.h"
#import "WKObject.h"
#import <wtf/OptionSet.h>

namespace WebKit {

template<> struct WrapperTraits<API::WebsiteDataRecord> {
    using WrapperClass = WKWebsiteDataRecord;
};

static inline std::optional<WebsiteDataType> toWebsiteDataType(NSString *websiteDataType)
{
    if ([websiteDataType isEqualToString:WKWebsiteDataTypeCookies])
        return WebsiteDataType::Cookies;
    if ([websiteDataType isEqualToString:WKWebsiteDataTypeFetchCache])
        return WebsiteDataType::DOMCache;
    if ([websiteDataType isEqualToString:WKWebsiteDataTypeDiskCache])
        return WebsiteDataType::DiskCache;
    if ([websiteDataType isEqualToString:WKWebsiteDataTypeMemoryCache])
        return WebsiteDataType::MemoryCache;
    if ([websiteDataType isEqualToString:WKWebsiteDataTypeOfflineWebApplicationCache])
        return WebsiteDataType::OfflineWebApplicationCache;
    if ([websiteDataType isEqualToString:WKWebsiteDataTypeSessionStorage])
        return WebsiteDataType::SessionStorage;
    if ([websiteDataType isEqualToString:WKWebsiteDataTypeLocalStorage])
        return WebsiteDataType::LocalStorage;
    if ([websiteDataType isEqualToString:WKWebsiteDataTypeWebSQLDatabases])
        return WebsiteDataType::WebSQLDatabases;
    if ([websiteDataType isEqualToString:WKWebsiteDataTypeIndexedDBDatabases])
        return WebsiteDataType::IndexedDBDatabases;
    if ([websiteDataType isEqualToString:WKWebsiteDataTypeServiceWorkerRegistrations])
        return WebsiteDataType::ServiceWorkerRegistrations;
    if ([websiteDataType isEqualToString:WKWebsiteDataTypeFileSystem])
        return WebsiteDataType::FileSystem;
    if ([websiteDataType isEqualToString:_WKWebsiteDataTypeHSTSCache])
        return WebsiteDataType::HSTSCache;
    if ([websiteDataType isEqualToString:WKWebsiteDataTypeMediaKeys])
        return WebsiteDataType::MediaKeys;
    if ([websiteDataType isEqualToString:WKWebsiteDataTypeHashSalt])
        return WebsiteDataType::DeviceIdHashSalt;
    if ([websiteDataType isEqualToString:WKWebsiteDataTypeSearchFieldRecentSearches])
        return WebsiteDataType::SearchFieldRecentSearches;
    if ([websiteDataType isEqualToString:_WKWebsiteDataTypeResourceLoadStatistics])
        return WebsiteDataType::ResourceLoadStatistics;
    if ([websiteDataType isEqualToString:_WKWebsiteDataTypeCredentials])
        return WebsiteDataType::Credentials;
    if ([websiteDataType isEqualToString:_WKWebsiteDataTypeAdClickAttributions])
        return WebsiteDataType::PrivateClickMeasurements;
    if ([websiteDataType isEqualToString:_WKWebsiteDataTypePrivateClickMeasurements])
        return WebsiteDataType::PrivateClickMeasurements;
#if HAVE(ALTERNATIVE_SERVICE)
    if ([websiteDataType isEqualToString:_WKWebsiteDataTypeAlternativeServices])
        return WebsiteDataType::AlternativeServices;
#endif
    return std::nullopt;
}

static inline OptionSet<WebKit::WebsiteDataType> toWebsiteDataTypes(NSSet *websiteDataTypes)
{
    OptionSet<WebKit::WebsiteDataType> result;

    for (NSString *websiteDataType in websiteDataTypes) {
        if (auto dataType = toWebsiteDataType(websiteDataType))
            result.add(*dataType);
    }

    return result;
}

static inline RetainPtr<NSSet> toWKWebsiteDataTypes(OptionSet<WebKit::WebsiteDataType> websiteDataTypes)
{
    auto wkWebsiteDataTypes = adoptNS([[NSMutableSet alloc] init]);

    if (websiteDataTypes.contains(WebsiteDataType::Cookies))
        [wkWebsiteDataTypes addObject:WKWebsiteDataTypeCookies];
    if (websiteDataTypes.contains(WebsiteDataType::DiskCache))
        [wkWebsiteDataTypes addObject:WKWebsiteDataTypeDiskCache];
    if (websiteDataTypes.contains(WebsiteDataType::DOMCache))
        [wkWebsiteDataTypes addObject:WKWebsiteDataTypeFetchCache];
    if (websiteDataTypes.contains(WebsiteDataType::MemoryCache))
        [wkWebsiteDataTypes addObject:WKWebsiteDataTypeMemoryCache];
    if (websiteDataTypes.contains(WebsiteDataType::OfflineWebApplicationCache))
        [wkWebsiteDataTypes addObject:WKWebsiteDataTypeOfflineWebApplicationCache];
    if (websiteDataTypes.contains(WebsiteDataType::SessionStorage))
        [wkWebsiteDataTypes addObject:WKWebsiteDataTypeSessionStorage];
    if (websiteDataTypes.contains(WebsiteDataType::LocalStorage))
        [wkWebsiteDataTypes addObject:WKWebsiteDataTypeLocalStorage];
    if (websiteDataTypes.contains(WebsiteDataType::WebSQLDatabases))
        [wkWebsiteDataTypes addObject:WKWebsiteDataTypeWebSQLDatabases];
    if (websiteDataTypes.contains(WebsiteDataType::IndexedDBDatabases))
        [wkWebsiteDataTypes addObject:WKWebsiteDataTypeIndexedDBDatabases];
    if (websiteDataTypes.contains(WebsiteDataType::ServiceWorkerRegistrations))
        [wkWebsiteDataTypes addObject:WKWebsiteDataTypeServiceWorkerRegistrations];
    if (websiteDataTypes.contains(WebsiteDataType::FileSystem))
        [wkWebsiteDataTypes addObject:WKWebsiteDataTypeFileSystem];
    if (websiteDataTypes.contains(WebsiteDataType::HSTSCache))
        [wkWebsiteDataTypes addObject:_WKWebsiteDataTypeHSTSCache];
    if (websiteDataTypes.contains(WebsiteDataType::MediaKeys))
        [wkWebsiteDataTypes addObject:WKWebsiteDataTypeMediaKeys];
    if (websiteDataTypes.contains(WebsiteDataType::SearchFieldRecentSearches))
        [wkWebsiteDataTypes addObject:WKWebsiteDataTypeSearchFieldRecentSearches];
    if (websiteDataTypes.contains(WebsiteDataType::DeviceIdHashSalt))
        [wkWebsiteDataTypes addObject:WKWebsiteDataTypeHashSalt];
    if (websiteDataTypes.contains(WebsiteDataType::ResourceLoadStatistics))
        [wkWebsiteDataTypes addObject:_WKWebsiteDataTypeResourceLoadStatistics];
    if (websiteDataTypes.contains(WebsiteDataType::Credentials))
        [wkWebsiteDataTypes addObject:_WKWebsiteDataTypeCredentials];
    if (websiteDataTypes.contains(WebsiteDataType::PrivateClickMeasurements))
        [wkWebsiteDataTypes addObject:_WKWebsiteDataTypePrivateClickMeasurements];
#if HAVE(ALTERNATIVE_SERVICE)
    if (websiteDataTypes.contains(WebsiteDataType::AlternativeServices))
        [wkWebsiteDataTypes addObject:_WKWebsiteDataTypeAlternativeServices];
#endif

    return wkWebsiteDataTypes;
}

}

@interface WKWebsiteDataRecord () <WKObject> {
@package
    API::ObjectStorage<API::WebsiteDataRecord> _websiteDataRecord;
}
@end
