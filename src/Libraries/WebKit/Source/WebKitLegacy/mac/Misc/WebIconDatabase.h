/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 10, 2025.
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
#if TARGET_OS_IPHONE
#import <Foundation/Foundation.h>
#else
#import <Cocoa/Cocoa.h>
#endif

#define ICON_DATABASE_DEPRECATED __deprecated_msg("WebIconDatabase is not API and no longer handles icon loading. It will be removed in a future release.")

#if !defined(WK_UNUSED_INSTANCE_VARIABLE)
#define WK_UNUSED_INSTANCE_VARIABLE
#endif

extern NSString *WebIconDatabaseDidAddIconNotification;
extern NSString *WebIconDatabaseDidRemoveAllIconsNotification;

extern NSString *WebIconNotificationUserInfoURLKey;

extern NSString *WebIconDatabaseDirectoryDefaultsKey;

extern NSSize WebIconSmallSize;  // 16 x 16
extern NSSize WebIconMediumSize; // 32 x 32
extern NSSize WebIconLargeSize;  // 128 x 128

@class WebIconDatabasePrivate;

/*!
    @class WebIconDatabase
    @discussion Features:
        - memory cache icons at different sizes
        - disk storage
        - icon update notification
        
        Uses:
        - UI elements to retrieve icons that represent site URLs.
        - Save icons to disk for later use.
 
    Every icon in the database has a retain count.  If an icon has a retain count greater than 0, it will be written to disk for later use. If an icon's retain count equals zero it will be removed from disk.  The retain count is not persistent across launches. If the WebKit client wishes to retain an icon it should retain the icon once for every launch.  This is best done at initialization time before the database begins removing icons.  To make sure that the database does not remove unretained icons prematurely, call delayDatabaseCleanup until all desired icons are retained.  Once all are retained, call allowDatabaseCleanup.
    
    Note that an icon can be retained after the database clean-up has begun. This just has to be done before the icon is removed. Icons are removed from the database whenever new icons are added to it.
    
    Retention methods can be called for icons that are not yet in the database.
*/

ICON_DATABASE_DEPRECATED
@interface WebIconDatabase : NSObject {

@private
    WK_UNUSED_INSTANCE_VARIABLE WebIconDatabasePrivate *_private;
}


/*!
    @method sharedIconDatabase
    @abstract Returns a shared instance of the icon database
*/
+ (WebIconDatabase *)sharedIconDatabase ICON_DATABASE_DEPRECATED;

#if !TARGET_OS_IPHONE
/*!
    @method iconForURL:withSize:
    @discussion Calls iconForURL:withSize:cache: with YES for cache.
*/
- (NSImage *)iconForURL:(NSString *)URL withSize:(NSSize)size ICON_DATABASE_DEPRECATED;

/*!
    @method iconForURL:withSize:cache:
    @discussion Returns an icon for a web site URL from memory or disk. nil if none is found.
    Usually called by a UI element to determine if a site URL has an associated icon.
    Often called by the observer of WebIconChangedNotification after the notification is sent.
    @param cache If yes, caches the returned image in memory if not already cached
*/
- (NSImage *)iconForURL:(NSString *)URL withSize:(NSSize)size cache:(BOOL)cache ICON_DATABASE_DEPRECATED;
#endif

/*!
    @method iconURLForURL:withSize:cache:
    @discussion Returns an icon URL for a web site URL from memory or disk. nil if none is found.
*/
- (NSString *)iconURLForURL:(NSString *)URL ICON_DATABASE_DEPRECATED;

#if !TARGET_OS_IPHONE
/*!
    @method defaultIconWithSize:
*/
- (NSImage *)defaultIconWithSize:(NSSize)size ICON_DATABASE_DEPRECATED;
- (NSImage *)defaultIconForURL:(NSString *)URL withSize:(NSSize)size ICON_DATABASE_DEPRECATED;
#endif

/*!
    @method retainIconForURL:
    @abstract Increments the retain count of the icon.
*/
- (void)retainIconForURL:(NSString *)URL ICON_DATABASE_DEPRECATED;

/*!
    @method releaseIconForURL:
    @abstract Decrements the retain count of the icon.
*/
- (void)releaseIconForURL:(NSString *)URL ICON_DATABASE_DEPRECATED;

/*!
    @method delayDatabaseCleanup:
    @discussion Only effective if called before the database begins removing icons.
    delayDatabaseCleanUp increments an internal counter that when 0 begins the database clean-up.
    The counter equals 0 at initialization.
*/
+ (void)delayDatabaseCleanup ICON_DATABASE_DEPRECATED;

/*!
    @method allowDatabaseCleanup:
    @discussion Informs the database that it now can begin removing icons.
    allowDatabaseCleanup decrements an internal counter that when 0 begins the database clean-up.
    The counter equals 0 at initialization.
*/
+ (void)allowDatabaseCleanup ICON_DATABASE_DEPRECATED;

- (void)setDelegate:(id)delegate ICON_DATABASE_DEPRECATED;
- (id)delegate ICON_DATABASE_DEPRECATED;

@end
