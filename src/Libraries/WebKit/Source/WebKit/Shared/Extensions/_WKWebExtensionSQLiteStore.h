/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 17, 2025.
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
#pragma once

#import "_WKWebExtensionSQLiteDatabase.h"

NS_ASSUME_NONNULL_BEGIN

@interface _WKWebExtensionSQLiteStore : NSObject {
@protected
    NSString *_uniqueIdentifier;
    NSURL *_directory;
    _WKWebExtensionSQLiteDatabase *_database;
    dispatch_queue_t _databaseQueue;
    BOOL _useInMemoryDatabase;
}

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

- (instancetype)initWithUniqueIdentifier:(NSString *)uniqueIdentifier directory:(NSString *)directory usesInMemoryDatabase:(BOOL)useInMemoryDatabase;

@property (nonatomic, readonly) BOOL useInMemoryDatabase;

- (void)close;

- (void)deleteDatabaseWithCompletionHandler:(void (^)(NSString * _Nullable errorMessage))completionHandler;

- (SchemaVersion)_migrateToCurrentSchemaVersionIfNeeded;

- (SchemaVersion)_databaseSchemaVersion;
- (DatabaseResult)_setDatabaseSchemaVersion:(SchemaVersion)newVersion;

- (BOOL)_openDatabaseIfNecessaryReturningErrorMessage:(NSString * _Nullable * _Nonnull)outErrorMessage;
- (BOOL)_openDatabaseIfNecessaryReturningErrorMessage:(NSString * _Nullable * _Nonnull)outErrorMessage createIfNecessary:(BOOL)createIfNecessary;

- (nullable NSString *)_deleteDatabaseIfEmpty;

- (void)createSavepointWithCompletionHandler:(void (^)(NSUUID * _Nullable savepointIdentifer, NSString * _Nullable errorMessage))completionHandler;
- (void)commitSavepoint:(NSUUID *)savepointIdentifier completionHandler:(void (^)(NSString * _Nullable errorMessage))completionHandler;
- (void)rollbackToSavepoint:(NSUUID *)savepointIdentifier completionHandler:(void (^)(NSString * _Nullable errorMessage))completionHandler;

@end

NS_ASSUME_NONNULL_END
