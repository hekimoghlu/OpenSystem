/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 21, 2024.
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
#import "WKWebsiteDataRecordInternal.h"

#import "_WKWebsiteDataSizeInternal.h"
#import <WebCore/SecurityOriginData.h>
#import <WebCore/WebCoreObjCExtras.h>

NSString * const WKWebsiteDataTypeFetchCache = @"WKWebsiteDataTypeFetchCache";
NSString * const WKWebsiteDataTypeDiskCache = @"WKWebsiteDataTypeDiskCache";
NSString * const WKWebsiteDataTypeMemoryCache = @"WKWebsiteDataTypeMemoryCache";
NSString * const WKWebsiteDataTypeOfflineWebApplicationCache = @"WKWebsiteDataTypeOfflineWebApplicationCache";

NSString * const WKWebsiteDataTypeCookies = @"WKWebsiteDataTypeCookies";
NSString * const WKWebsiteDataTypeSessionStorage = @"WKWebsiteDataTypeSessionStorage";

NSString * const WKWebsiteDataTypeLocalStorage = @"WKWebsiteDataTypeLocalStorage";
NSString * const WKWebsiteDataTypeWebSQLDatabases = @"WKWebsiteDataTypeWebSQLDatabases";
NSString * const WKWebsiteDataTypeIndexedDBDatabases = @"WKWebsiteDataTypeIndexedDBDatabases";
NSString * const WKWebsiteDataTypeServiceWorkerRegistrations = @"WKWebsiteDataTypeServiceWorkerRegistrations";
NSString * const WKWebsiteDataTypeFileSystem = @"WKWebsiteDataTypeFileSystem";
NSString * const WKWebsiteDataTypeSearchFieldRecentSearches = @"WKWebsiteDataTypeSearchFieldRecentSearches";
NSString * const WKWebsiteDataTypeMediaKeys = @"WKWebsiteDataTypeMediaKeys";
NSString * const WKWebsiteDataTypeHashSalt = @"WKWebsiteDataTypeHashSalt";

NSString * const _WKWebsiteDataTypeMediaKeys = WKWebsiteDataTypeMediaKeys;
NSString * const _WKWebsiteDataTypeHSTSCache = @"_WKWebsiteDataTypeHSTSCache";
NSString * const _WKWebsiteDataTypeSearchFieldRecentSearches = WKWebsiteDataTypeSearchFieldRecentSearches;
NSString * const _WKWebsiteDataTypeResourceLoadStatistics = @"_WKWebsiteDataTypeResourceLoadStatistics";
NSString * const _WKWebsiteDataTypeCredentials = @"_WKWebsiteDataTypeCredentials";
NSString * const _WKWebsiteDataTypeAdClickAttributions = @"_WKWebsiteDataTypeAdClickAttributions";
NSString * const _WKWebsiteDataTypePrivateClickMeasurements = @"_WKWebsiteDataTypePrivateClickMeasurements";
NSString * const _WKWebsiteDataTypeAlternativeServices = @"_WKWebsiteDataTypeAlternativeServices";
NSString * const _WKWebsiteDataTypeFileSystem = WKWebsiteDataTypeFileSystem;

@implementation WKWebsiteDataRecord

WK_OBJECT_DISABLE_DISABLE_KVC_IVAR_ACCESS;

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainRunLoop(WKWebsiteDataRecord.class, self))
        return;

    _websiteDataRecord->API::WebsiteDataRecord::~WebsiteDataRecord();

    [super dealloc];
}

static NSString *dataTypesToString(NSSet *dataTypes)
{
    auto array = adoptNS([[NSMutableArray alloc] init]);

    if ([dataTypes containsObject:WKWebsiteDataTypeDiskCache])
        [array addObject:@"Disk Cache"];
    if ([dataTypes containsObject:WKWebsiteDataTypeFetchCache])
        [array addObject:@"Fetch Cache"];
    if ([dataTypes containsObject:WKWebsiteDataTypeMemoryCache])
        [array addObject:@"Memory Cache"];
    if ([dataTypes containsObject:WKWebsiteDataTypeOfflineWebApplicationCache])
        [array addObject:@"Offline Web Application Cache"];
    if ([dataTypes containsObject:WKWebsiteDataTypeCookies])
        [array addObject:@"Cookies"];
    if ([dataTypes containsObject:WKWebsiteDataTypeSessionStorage])
        [array addObject:@"Session Storage"];
    if ([dataTypes containsObject:WKWebsiteDataTypeLocalStorage])
        [array addObject:@"Local Storage"];
    if ([dataTypes containsObject:WKWebsiteDataTypeWebSQLDatabases])
        [array addObject:@"Web SQL"];
    if ([dataTypes containsObject:WKWebsiteDataTypeIndexedDBDatabases])
        [array addObject:@"IndexedDB"];
    if ([dataTypes containsObject:WKWebsiteDataTypeServiceWorkerRegistrations])
        [array addObject:@"Service Worker Registrations"];
    if ([dataTypes containsObject:_WKWebsiteDataTypeHSTSCache])
        [array addObject:@"HSTS Cache"];
    if ([dataTypes containsObject:WKWebsiteDataTypeMediaKeys])
        [array addObject:@"Media Keys"];
    if ([dataTypes containsObject:WKWebsiteDataTypeHashSalt])
        [array addObject:@"Hash Salt"];
    if ([dataTypes containsObject:WKWebsiteDataTypeSearchFieldRecentSearches])
        [array addObject:@"Search Field Recent Searches"];
    if ([dataTypes containsObject:WKWebsiteDataTypeFileSystem])
        [array addObject:@"File System"];
    if ([dataTypes containsObject:_WKWebsiteDataTypeResourceLoadStatistics])
        [array addObject:@"Resource Load Statistics"];
    if ([dataTypes containsObject:_WKWebsiteDataTypeCredentials])
        [array addObject:@"Credentials"];
    if ([dataTypes containsObject:_WKWebsiteDataTypeAdClickAttributions] || [dataTypes containsObject:_WKWebsiteDataTypePrivateClickMeasurements])
        [array addObject:@"Private Click Measurements"];
    if ([dataTypes containsObject:_WKWebsiteDataTypeAlternativeServices])
        [array addObject:@"Alternative Services"];

    return [array componentsJoinedByString:@", "];
}

- (NSString *)description
{
    auto result = adoptNS([[NSMutableString alloc] initWithFormat:@"<%@: %p; displayName = %@; dataTypes = { %@ }", NSStringFromClass(self.class), self, self.displayName, dataTypesToString(self.dataTypes)]);

    if (auto* dataSize = self._dataSize)
        [result appendFormat:@"; _dataSize = { %llu bytes }", dataSize.totalSize];

    [result appendString:@">"];
    return result.autorelease();
}

- (NSString *)displayName
{
    return _websiteDataRecord->websiteDataRecord().displayName;
}

- (NSSet *)dataTypes
{
    return WebKit::toWKWebsiteDataTypes(_websiteDataRecord->websiteDataRecord().types).autorelease();
}

#pragma mark WKObject protocol implementation

- (API::Object&)_apiObject
{
    return *_websiteDataRecord;
}

@end

@implementation WKWebsiteDataRecord (WKPrivate)

- (_WKWebsiteDataSize *)_dataSize
{
    auto& size = _websiteDataRecord->websiteDataRecord().size;

    if (!size)
        return nil;

    return adoptNS([[_WKWebsiteDataSize alloc] initWithSize:*size]).autorelease();
}

- (NSArray<NSString *> *)_originsStrings
{
    return createNSArray(_websiteDataRecord->websiteDataRecord().origins, [] (auto& origin) -> NSString * {
        return origin.toString();
    }).autorelease();
}

@end
