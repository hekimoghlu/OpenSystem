/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 30, 2021.
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
#import <WebKit/WKFoundation.h>

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/*! @constant WKWebsiteDataTypeFetchCache On-disk Fetch caches. */
WK_EXTERN NSString * const WKWebsiteDataTypeFetchCache WK_API_AVAILABLE(macos(10.13.4), ios(11.3));

/*! @constant WKWebsiteDataTypeDiskCache On-disk caches. */
WK_EXTERN NSString * const WKWebsiteDataTypeDiskCache WK_API_AVAILABLE(macos(10.11), ios(9.0));

/*! @constant WKWebsiteDataTypeMemoryCache In-memory caches. */
WK_EXTERN NSString * const WKWebsiteDataTypeMemoryCache WK_API_AVAILABLE(macos(10.11), ios(9.0));

/*! @constant WKWebsiteDataTypeOfflineWebApplicationCache HTML offline web application caches. */
WK_EXTERN NSString * const WKWebsiteDataTypeOfflineWebApplicationCache WK_API_AVAILABLE(macos(10.11), ios(9.0));

/*! @constant WKWebsiteDataTypeCookies Cookies. */
WK_EXTERN NSString * const WKWebsiteDataTypeCookies WK_API_AVAILABLE(macos(10.11), ios(9.0));

/*! @constant WKWebsiteDataTypeSessionStorage HTML session storage. */
WK_EXTERN NSString * const WKWebsiteDataTypeSessionStorage WK_API_AVAILABLE(macos(10.11), ios(9.0));

/*! @constant WKWebsiteDataTypeLocalStorage HTML local storage. */
WK_EXTERN NSString * const WKWebsiteDataTypeLocalStorage WK_API_AVAILABLE(macos(10.11), ios(9.0));

/*! @constant WKWebsiteDataTypeWebSQLDatabases WebSQL databases. */
WK_EXTERN NSString * const WKWebsiteDataTypeWebSQLDatabases WK_API_AVAILABLE(macos(10.11), ios(9.0));

/*! @constant WKWebsiteDataTypeIndexedDBDatabases IndexedDB databases. */
WK_EXTERN NSString * const WKWebsiteDataTypeIndexedDBDatabases WK_API_AVAILABLE(macos(10.11), ios(9.0));

/*! @constant WKWebsiteDataTypeServiceWorkerRegistrations Service worker registrations. */
WK_EXTERN NSString * const WKWebsiteDataTypeServiceWorkerRegistrations WK_API_AVAILABLE(macos(10.13.4), ios(11.3));

/*! @constant WKWebsiteDataTypeFileSystem File system storage. */
WK_EXTERN NSString * const WKWebsiteDataTypeFileSystem WK_API_AVAILABLE(macos(13.0), ios(16.0));

/*! @constant WKWebsiteDataTypeSearchFieldRecentSearches Search field history. */
WK_EXTERN NSString * const WKWebsiteDataTypeSearchFieldRecentSearches WK_API_AVAILABLE(macos(14.0), ios(17.0));

/*! @constant WKWebsiteDataTypeMediaKeys MediaKeys storage */
WK_EXTERN NSString * const WKWebsiteDataTypeMediaKeys WK_API_AVAILABLE(macos(14.0), ios(17.0));

/*! @constant WKWebsiteDataTypeHashSalt Hash salt for deviceId */
WK_EXTERN NSString * const WKWebsiteDataTypeHashSalt WK_API_AVAILABLE(macos(14.0), ios(17.0));

/*! A WKWebsiteDataRecord represents website data, grouped by domain name using the public suffix list. */
WK_SWIFT_UI_ACTOR
WK_CLASS_AVAILABLE(macos(10.11), ios(9.0))
@interface WKWebsiteDataRecord : NSObject

/*! @abstract The display name for the data record. This is usually the domain name. */
@property (nonatomic, readonly, copy) NSString *displayName;

/*! @abstract The various types of website data that exist for this data record. */
@property (nonatomic, readonly, copy) NSSet<NSString *> *dataTypes;

@end

NS_ASSUME_NONNULL_END
