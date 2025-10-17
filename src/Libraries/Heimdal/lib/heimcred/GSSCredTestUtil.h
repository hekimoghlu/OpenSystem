/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 18, 2024.
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
#import <XCTest/XCTest.h>
#import "hc_err.h"
#import "common.h"
#import "heimbase.h"
#import "MockManagedAppManager.h"

NS_ASSUME_NONNULL_BEGIN

@interface GSSCredTestUtil : NSObject

#pragma mark -
#pragma mark peer

+ (struct peer *)createPeer:(NSString *)bundle identifier:(int)sid;
+ (struct peer *)createPeer:(NSString *)bundle callingBundleId:(NSString * _Nullable)callingApp identifier:(int)sid;
+ (void)freePeer:(struct peer * _Nullable)ptr;

#pragma mark -
#pragma mark create

+ (BOOL)createCredentialAndCache:(struct peer * _Nullable)peer name:(NSString*)clientName returningCacheUuid:(CFUUIDRef _Nonnull *_Nonnull)uuid;
+ (BOOL)createCredentialAndCache:(struct peer * _Nullable)peer name:(NSString*)clientName returningCacheUuid:(CFUUIDRef *)cacheUUID credentialUUID:(CFUUIDRef *)credUUID;
+ (BOOL)createCredentialAndCache:(struct peer * _Nullable)peer name:(NSString*)clientName returningCredentialDictionary:(CFDictionaryRef _Nonnull *_Nonnull)dict;

+ (BOOL)createCredential:(struct peer * _Nullable)peer name:(NSString*)clientName attributes:(CFDictionaryRef  _Nullable)attributes returningUuid:(CFUUIDRef _Nonnull *_Nonnull) uuid;
+ (BOOL)createCredential:(struct peer * _Nullable)peer name:(NSString*)clientName attributes:(CFDictionaryRef _Nullable)attributes returningDictionary:(CFDictionaryRef _Nonnull *_Nonnull)dict;
+ (BOOL)executeCreateCred:(struct peer * _Nullable)peer forAttributes:(CFDictionaryRef)allAttrs returningDictionary:(CFDictionaryRef _Nullable * _Nonnull)dict;

+ (BOOL)createNTLMCredential:(struct peer * _Nullable)peer returningUuid:(CFUUIDRef _Nonnull *_Nonnull)uuid;
+ (BOOL)createNTLMCredential:(struct peer * _Nullable)peer returningDictionary:(CFDictionaryRef _Nonnull *_Nonnull)dict;
+ (BOOL)createNTLMCredential:(struct peer * _Nullable)peer attributes:(CFDictionaryRef _Nullable)attributes returningDictionary:(CFDictionaryRef _Nullable *_Nonnull)dict;

+ (BOOL)addNTLMChallenge:(struct peer * _Nullable)peer challenge:(uint8_t [8])challenge;
+ (BOOL)checkNTLMChallenge:(struct peer * _Nullable)peer challenge:(uint8_t [8])challenge;

#pragma mark -
#pragma mark move

+ (BOOL)move:(struct peer * _Nullable)peer from:(CFUUIDRef)from to:(CFUUIDRef)to;

#pragma mark -
#pragma mark fetch

+ (BOOL)fetchCredential:(struct peer * _Nullable)peer uuid:(CFUUIDRef)uuid;
+ (BOOL)fetchCredential:(struct peer * _Nullable)peer uuid:(CFUUIDRef) uuid returningDictionary:(CFDictionaryRef _Nonnull *_Nonnull)dict;
+ (CFUUIDRef _Nullable)getDefaultCredential:(struct peer * _Nullable)peer CF_RETURNS_RETAINED;
+ (BOOL)fetchDefaultCredential:(struct peer * _Nullable)peer returningName:(CFStringRef * _Nonnull)name;

#pragma mark -
#pragma mark query

+ (NSUInteger)itemCount:(struct peer * _Nullable)peer;
+ (BOOL)queryAllKerberos:(struct peer * _Nullable)peer returningArray:(NSArray * _Nonnull __autoreleasing *_Nonnull)items;
+ (BOOL)queryAll:(struct peer * _Nullable)peer parentUUID:(CFUUIDRef)parentUUID returningArray:(NSArray * _Nonnull __autoreleasing *_Nonnull)items;
+ (BOOL)queryAll:(struct peer * _Nullable)peer type:(CFStringRef)type returningArray:(NSArray * _Nonnull __autoreleasing *_Nonnull)items;
+ (BOOL)queryAllCredentials:(struct peer * _Nullable)peer returningArray:(NSArray * _Nonnull __autoreleasing *_Nonnull)items;
+ (void)showStatus:(struct peer * _Nullable)peer;

#pragma mark -
#pragma mark update

+ (int64_t)setAttributes:(struct peer * _Nullable)peer uuid:(CFUUIDRef) uuid attributes:(CFDictionaryRef _Nonnull)attributes returningDictionary:(CFDictionaryRef _Nonnull * _Nullable)dict;

#pragma mark -
#pragma mark delete

+ (int64_t)delete:(struct peer * _Nullable)peer uuid:(CFUUIDRef)uuid;
+ (int64_t)deleteAll:(struct peer * _Nullable)peer dsid:(NSString *)dsid;
+ (BOOL)deleteCacheContents:(struct peer * _Nullable)peer parentUUID:(CFUUIDRef)parentUUID;

#pragma mark -
#pragma mark hold

+ (int64_t)hold:(struct peer * _Nullable)peer uuid:(CFUUIDRef)uuid;
+ (int64_t)unhold:(struct peer * _Nullable)peer uuid:(CFUUIDRef)uuid;

#pragma mark -
#pragma mark utility

+ (void)flushCache;

@end

NS_ASSUME_NONNULL_END
