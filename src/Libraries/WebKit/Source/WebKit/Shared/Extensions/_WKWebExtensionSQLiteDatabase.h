/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 8, 2023.
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

NS_ASSUME_NONNULL_BEGIN

typedef int DatabaseResult;
typedef int _WKSQLiteErrorCode;
typedef int SchemaVersion;

typedef NS_ENUM(NSInteger, SQLiteDatabaseAccessType) {
    SQLiteDatabaseAccessTypeReadOnly,
    SQLiteDatabaseAccessTypeReadWrite,
    SQLiteDatabaseAccessTypeReadWriteCreate,
};

// This enum is only applicable on iOS and has no effect on macOS.
// SQLiteDatabaseProtectionTypeDefault sets the protection to class C.
typedef NS_ENUM(NSInteger, SQLiteDatabaseProtectionType) {
    SQLiteDatabaseProtectionTypeDefault,
    SQLiteDatabaseProtectionTypeCompleteUntilFirstUserAuthentication,
    SQLiteDatabaseProtectionTypeCompleteUnlessOpen,
    SQLiteDatabaseProtectionTypeComplete,
};

struct sqlite3;
struct sqlite3_stmt;

extern NSString * const _WKWebExtensionSQLiteErrorDomain;

@interface _WKWebExtensionSQLiteDatabase : NSObject

@property (readonly, nonatomic, nullable) sqlite3 *handle;
@property (readonly, nonatomic) NSURL *url;

@property (readonly, nonatomic) _WKSQLiteErrorCode lastErrorCode;
@property (readonly, nonatomic, nullable) NSString *lastErrorMessage;

@property (readonly, nonatomic) dispatch_queue_t queue;

- (instancetype)initWithURL:(NSURL *)url queue:(dispatch_queue_t)queue;

- (BOOL)enableWAL:(NSError * __autoreleasing  * _Nullable)error;

- (BOOL)openWithAccessType:(SQLiteDatabaseAccessType)accessType error:(NSError **)error;
- (BOOL)openWithAccessType:(SQLiteDatabaseAccessType)accessType vfs:(nullable NSString *)vfs error:(NSError **)error;
- (BOOL)openWithAccessType:(SQLiteDatabaseAccessType)accessType protectionType:(SQLiteDatabaseProtectionType)protectionType vfs:(nullable NSString *)vfs error:(NSError **)error;

- (BOOL)reportErrorWithCode:(_WKSQLiteErrorCode)errorCode query:(nullable NSString *)query error:(NSError **)error;
- (BOOL)reportErrorWithCode:(_WKSQLiteErrorCode)errorCode statement:(sqlite3_stmt*)statement error:(NSError **)error;

- (_WKSQLiteErrorCode)close;

+ (NSURL *)inMemoryDatabaseURL;

@end

NS_ASSUME_NONNULL_END
