/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 20, 2022.
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
#import <Foundation/Foundation.h>

extern NSString *WebDatabaseDirectoryDefaultsKey;

extern NSString *WebDatabaseDisplayNameKey;
extern NSString *WebDatabaseExpectedSizeKey;
extern NSString *WebDatabaseUsageKey;

// Posted with an origin is created from scratch, gets a new database, has a database deleted, has a quota change, etc
// The notification object will be a WebSecurityOrigin object corresponding to the origin.
extern NSString *WebDatabaseDidModifyOriginNotification;

// Posted when a database is created, its size increases, its display name changes, or its estimated size changes, or the database is removed
// The notification object will be a WebSecurityOrigin object corresponding to the origin.
// The notification userInfo will have a WebDatabaseNameKey whose value is the database name.
extern NSString *WebDatabaseDidModifyDatabaseNotification;
extern NSString *WebDatabaseIdentifierKey;

#if TARGET_OS_IPHONE
#import <WebKitLegacy/WebUIKitSupport.h>

// Posted when origins have changed.
extern CFStringRef WebDatabaseOriginsDidChangeNotification;
#endif

@class WebSecurityOrigin;

@interface WebDatabaseManager : NSObject

+ (WebDatabaseManager *)sharedWebDatabaseManager;

// Will return an array of WebSecurityOrigin objects.
- (NSArray *)origins;

// Will return an array of strings, the identifiers of each database in the given origin.
- (NSArray *)databasesWithOrigin:(WebSecurityOrigin *)origin;

// Will return the dictionary describing everything about the database for the passed identifier and origin.
- (NSDictionary *)detailsForDatabase:(NSString *)databaseIdentifier withOrigin:(WebSecurityOrigin *)origin;

- (void)deleteAllDatabases; // Deletes all databases and all origins.
- (BOOL)deleteOrigin:(WebSecurityOrigin *)origin;
- (BOOL)deleteDatabase:(NSString *)databaseIdentifier withOrigin:(WebSecurityOrigin *)origin;

// For DumpRenderTree support only
- (void)deleteAllIndexedDatabases;

#if TARGET_OS_IPHONE
+ (void)scheduleEmptyDatabaseRemoval;
#endif
@end
