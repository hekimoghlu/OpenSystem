/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 3, 2023.
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

@class WKHTTPCookieStore;

typedef NS_ENUM(NSInteger, WKCookiePolicy) {
    WKCookiePolicyAllow,
    WKCookiePolicyDisallow,
} NS_SWIFT_NAME(WKHTTPCookieStore.CookiePolicy) WK_API_AVAILABLE(macos(14.0), ios(17.0));

WK_API_AVAILABLE(macos(10.13), ios(11.0))
WK_SWIFT_UI_ACTOR
@protocol WKHTTPCookieStoreObserver <NSObject>
@optional
- (void)cookiesDidChangeInCookieStore:(WKHTTPCookieStore *)cookieStore;
@end

/*!
 A WKHTTPCookieStore object allows managing the HTTP cookies associated with a particular WKWebsiteDataStore.
 */
WK_CLASS_AVAILABLE(macos(10.13), ios(11.0))
WK_SWIFT_UI_ACTOR
@interface WKHTTPCookieStore : NSObject

- (instancetype)init NS_UNAVAILABLE;

/*! @abstract Fetches all stored cookies.
 @param completionHandler A block to invoke with the fetched cookies.
 */
- (void)getAllCookies:(WK_SWIFT_UI_ACTOR void (^)(NSArray<NSHTTPCookie *> *))completionHandler;

/*! @abstract Set a cookie.
 @param cookie The cookie to set.
 @param completionHandler A block to invoke once the cookie has been stored.
 */
- (void)setCookie:(NSHTTPCookie *)cookie completionHandler:(nullable WK_SWIFT_UI_ACTOR void (^)(void))completionHandler;

/*! @abstract Delete the specified cookie.
 @param completionHandler A block to invoke once the cookie has been deleted.
 */
- (void)deleteCookie:(NSHTTPCookie *)cookie completionHandler:(nullable WK_SWIFT_UI_ACTOR void (^)(void))completionHandler WK_SWIFT_ASYNC_NAME(deleteCookie(_:));

/*! @abstract Adds a WKHTTPCookieStoreObserver object with the cookie store.
 @param observer The observer object to add.
 @discussion The observer is not retained by the receiver. It is your responsibility
 to unregister the observer before it becomes invalid.
 */
- (void)addObserver:(id<WKHTTPCookieStoreObserver>)observer;

/*! @abstract Removes a WKHTTPCookieStoreObserver object from the cookie store.
 @param observer The observer to remove.
 */
- (void)removeObserver:(id<WKHTTPCookieStoreObserver>)observer;

/*! @abstract Set whether cookies are allowed.
  @param policy A value indicating whether cookies are allowed. The default value is WKCookiePolicyAllow.
  @param completionHandler A block to invoke once the cookie policy has been set.
  */
- (void)setCookiePolicy:(WKCookiePolicy)policy completionHandler:(nullable WK_SWIFT_UI_ACTOR void (^)(void))completionHandler WK_API_AVAILABLE(macos(14.0), ios(17.0));

/*! @abstract Get whether cookies are allowed.
 @param completionHandler A block to invoke with the value of whether cookies are allowed.
 */
- (void)getCookiePolicy:(WK_SWIFT_UI_ACTOR void (^)(WKCookiePolicy))completionHandler WK_SWIFT_ASYNC_NAME(getter:cookiePolicy()) WK_API_AVAILABLE(macos(14.0), ios(17.0));

@end

NS_ASSUME_NONNULL_END
