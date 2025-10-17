/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 4, 2023.
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

#import <WebKit/WKWebsiteDataRecord.h>

#if __has_include(<Network/proxy_config.h>)
#import <Network/Network.h>
#endif

NS_ASSUME_NONNULL_BEGIN

@class WKHTTPCookieStore;

/*! A WKWebsiteDataStore represents various types of data that a website might
 make use of. This includes cookies, disk and memory caches, and persistent data such as WebSQL,
 IndexedDB databases, and local storage.
 */
WK_SWIFT_UI_ACTOR
WK_CLASS_AVAILABLE(macos(10.11), ios(9.0))
@interface WKWebsiteDataStore : NSObject <NSSecureCoding>

/* @abstract Returns the default data store. */
+ (WKWebsiteDataStore *)defaultDataStore;

/** @abstract Returns a new non-persistent data store.
 @discussion If a WKWebView is associated with a non-persistent data store, no data will
 be written to the file system. This is useful for implementing "private browsing" in a web view.
*/
+ (WKWebsiteDataStore *)nonPersistentDataStore;

- (instancetype)new NS_UNAVAILABLE;
- (instancetype)init NS_UNAVAILABLE;

/*! @abstract Whether the data store is persistent or not. */
@property (nonatomic, readonly, getter=isPersistent) BOOL persistent;

/*! @abstract Returns a set of all available website data types. */
+ (NSSet<NSString *> *)allWebsiteDataTypes;

/*! @abstract Fetches data records containing the given website data types.
  @param dataTypes The website data types to fetch records for.
  @param completionHandler A block to invoke when the data records have been fetched.
*/
- (void)fetchDataRecordsOfTypes:(NSSet<NSString *> *)dataTypes completionHandler:(WK_SWIFT_UI_ACTOR void (^)(NSArray<WKWebsiteDataRecord *> *))completionHandler WK_SWIFT_ASYNC_NAME(dataRecords(ofTypes:));

/*! @abstract Removes website data of the given types for the given data records.
 @param dataTypes The website data types that should be removed.
 @param dataRecords The website data records to delete website data for.
 @param completionHandler A block to invoke when the website data for the records has been removed.
*/
- (void)removeDataOfTypes:(NSSet<NSString *> *)dataTypes forDataRecords:(NSArray<WKWebsiteDataRecord *> *)dataRecords completionHandler:(WK_SWIFT_UI_ACTOR void (^)(void))completionHandler;

/*! @abstract Removes all website data of the given types that has been modified since the given date.
 @param dataTypes The website data types that should be removed.
 @param date A date. All website data modified after this date will be removed.
 @param completionHandler A block to invoke when the website data has been removed.
*/
- (void)removeDataOfTypes:(NSSet<NSString *> *)dataTypes modifiedSince:(NSDate *)date completionHandler:(WK_SWIFT_UI_ACTOR void (^)(void))completionHandler;

/*! @abstract Returns the cookie store representing HTTP cookies in this website data store. */
@property (nonatomic, readonly) WKHTTPCookieStore *httpCookieStore WK_API_AVAILABLE(macos(10.13), ios(11.0));

/*! @abstract Get identifier for a data store.
 @discussion Returns nil for default and non-persistent data store .
 */
@property (nonatomic, readonly, nullable) NSUUID *identifier WK_API_AVAILABLE(macos(14.0), ios(17.0));

/*! @abstract Get a persistent data store.
 @param identifier An identifier that is used to uniquely identify the data store.
 @discussion If a data store with this identifier does not exist yet, it will be created. Throws exception if identifier
 is 0.
*/
+ (WKWebsiteDataStore *)dataStoreForIdentifier:(NSUUID *)identifier WK_API_AVAILABLE(macos(14.0), ios(17.0));

/*! @abstract Delete a persistent data store.
 @param identifier An identifier that is used to uniquely identify the data store.
 @param completionHandler A block to invoke with optional error when the operation completes.
 @discussion This should be called when the data store is not used any more. Returns error if removal fails
 to complete. WKWebView using the data store must be released before removal.
*/
+ (void)removeDataStoreForIdentifier:(NSUUID *)identifier completionHandler:(WK_SWIFT_UI_ACTOR void(^)(NSError * _Nullable))completionHandler WK_API_AVAILABLE(macos(14.0), ios(17.0));

/*! @abstract Fetch all data stores identifiers.
 @param completionHandler A block to invoke with an array of identifiers when the operation completes.
 @discussion Default or non-persistent data store do not have an identifier.
*/
+ (void)fetchAllDataStoreIdentifiers:(WK_SWIFT_UI_ACTOR void(^)(NSArray<NSUUID *> *))completionHandler WK_SWIFT_ASYNC_NAME(getter:allDataStoreIdentifiers()) WK_API_AVAILABLE(macos(14.0), ios(17.0));

#if ((TARGET_OS_OSX && __MAC_OS_X_VERSION_MAX_ALLOWED >= 140000) \
    || ((TARGET_OS_IOS || TARGET_OS_MACCATALYST) && __IPHONE_OS_VERSION_MAX_ALLOWED >= 170000) \
    || (TARGET_OS_WATCH && __WATCH_OS_VERSION_MAX_ALLOWED >= 100000) \
    || (TARGET_OS_TV && __TV_OS_VERSION_MAX_ALLOWED >= 170000) \
    || (defined(TARGET_OS_VISION) && TARGET_OS_VISION))
/*! @abstract Gets or sets the proxy configurations to be used to override networking in all WKWebViews that use this WKWebsiteDataStore.
 @discussion Changing the proxy configurations might interupt current networking operations in any WKWebView that use this WKWebsiteDataStore,
 so it is encouraged to finish setting the proxy configurations before starting any page loads.
*/
#if defined(OS_OBJECT_USE_OBJC) && OS_OBJECT_USE_OBJC
@property (nullable, nonatomic, copy) NSArray<nw_proxy_config_t> *proxyConfigurations NS_REFINED_FOR_SWIFT API_AVAILABLE(macos(14.0), ios(17.0));
#else
@property (nullable, nonatomic, copy) NSArray *proxyConfigurations NS_REFINED_FOR_SWIFT API_AVAILABLE(macos(14.0), ios(17.0));
#endif
#endif

@end

NS_ASSUME_NONNULL_END
