/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 27, 2025.
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
#if !__has_feature(objc_arc)
#error This file requires ARC. Add the "-fobjc-arc" compiler flag for this file.
#endif

#import "config.h"
#import "_WKWebExtensionSQLiteStatement.h"

#import "Logging.h"
#import "_WKWebExtensionSQLiteDatabase.h"
#import "_WKWebExtensionSQLiteHelpers.h"
#import "_WKWebExtensionSQLiteRow.h"
#import <sqlite3.h>

#if ENABLE(WK_WEB_EXTENSIONS)

using namespace WebKit;

@interface _WKWebExtensionSQLiteStatement ()
- (instancetype)init NS_DESIGNATED_INITIALIZER;
@end

@implementation _WKWebExtensionSQLiteStatement {
    _WKWebExtensionSQLiteDatabase *_database;
    sqlite3_stmt* _handle;

    NSDictionary *_columnNamesToIndexes;
    NSArray *_columnNames;
}

- (instancetype)init
{
    ASSERT_NOT_REACHED();
    return nil;
}

- (instancetype)initWithDatabase:(_WKWebExtensionSQLiteDatabase *)database query:(NSString *)query error:(NSError **)error
{
    ASSERT_ARG(database, database.handle);

    if (!(self = [super init]))
        return nil;

    _database = database;

    dispatch_assert_queue(_database.queue);
    _WKSQLiteErrorCode result = sqlite3_prepare_v2(database.handle, query.UTF8String, -1, &_handle, 0);
    if (result != SQLITE_OK) {
        [database reportErrorWithCode:result query:query error:error];
        return nil;
    }

    return self;
}

- (instancetype)initWithDatabase:(_WKWebExtensionSQLiteDatabase *)database query:(NSString *)query
{
    return [self initWithDatabase:database query:query error:nullptr];
}

- (void)dealloc
{
    sqlite3_stmt* handle = _handle;
    if (!handle)
        return;

    auto *database = _database;
    dispatch_async(database.queue, ^{
        // The database might have closed already.
        if (!database.handle)
            return;

        sqlite3_finalize(handle);
    });
}

- (_WKSQLiteErrorCode)execute
{
    dispatch_assert_queue(_database.queue);
    ASSERT(self._isValid);

    _WKSQLiteErrorCode resultCode = sqlite3_step(_handle);
    if (!SQLiteIsExecutionError(resultCode))
        return resultCode;

    [_database reportErrorWithCode:resultCode statement:_handle error:nullptr];
    return resultCode;
}

- (BOOL)execute:(NSError **)error
{
    dispatch_assert_queue(_database.queue);
    ASSERT(self._isValid);

    _WKSQLiteErrorCode resultCode = sqlite3_step(_handle);
    if (!SQLiteIsExecutionError(resultCode))
        return YES;

    [_database reportErrorWithCode:resultCode statement:_handle error:error];
    return NO;
}

- (_WKWebExtensionSQLiteRowEnumerator *)fetch
{
    dispatch_assert_queue(_database.queue);
    ASSERT(self._isValid);

    return [[_WKWebExtensionSQLiteRowEnumerator alloc] initWithResultsOfStatement:self];
}

- (BOOL)fetchWithEnumerationBlock:(void (^)(_WKWebExtensionSQLiteRow *row, BOOL* stop))enumerationBlock error:(NSError **)error
{
    dispatch_assert_queue(_database.queue);
    ASSERT(self._isValid);

    _WKWebExtensionSQLiteRow *row;

    _WKSQLiteErrorCode result = SQLITE_OK;
    BOOL stop = NO;
    while (!stop) {
        result = sqlite3_step(_handle);
        if (result != SQLITE_ROW)
            break;

        if (!row)
            row = [[_WKWebExtensionSQLiteRow alloc] initWithStatement:self];

        enumerationBlock(row, &stop);
    }

    if (result == SQLITE_DONE)
        return YES;

    return NO;
}

- (void)reset
{
    dispatch_assert_queue(_database.queue);
    ASSERT(self._isValid);

    _WKSQLiteErrorCode result = sqlite3_reset(_handle);
    if (result != SQLITE_OK)
        RELEASE_LOG_DEBUG(Extensions, "Could not reset statement: %@ (%d)", _database.lastErrorMessage, (int)result);
}

- (void)invalidate
{
    dispatch_assert_queue(_database.queue);
    ASSERT(self._isValid);

    _WKSQLiteErrorCode result = sqlite3_finalize(_handle);
    if (result != SQLITE_OK)
        RELEASE_LOG_DEBUG(Extensions, "Could not finalize statement: %@ (%d)", _database.lastErrorMessage, (int)result);
    _handle = nullptr;
}

- (void)bindString:(NSString *)string atParameterIndex:(NSUInteger)index
{
    dispatch_assert_queue(_database.queue);
    ASSERT(self._isValid);
    ASSERT_ARG(index, index > 0);

    _WKSQLiteErrorCode result = sqlite3_bind_text(_handle, index, string.UTF8String, -1, SQLITE_TRANSIENT);
    if (result != SQLITE_OK)
        RELEASE_LOG_DEBUG(Extensions, "Could not bind string: %@ (%d)", _database.lastErrorMessage, (int)result);
}

- (void)bindInt:(int)n atParameterIndex:(NSUInteger)index
{
    dispatch_assert_queue(_database.queue);
    ASSERT(self._isValid);
    ASSERT_ARG(index, index > 0);

    _WKSQLiteErrorCode result = sqlite3_bind_int(_handle, index, n);
    if (result != SQLITE_OK)
        RELEASE_LOG_DEBUG(Extensions, "Could not bind int: %@ (%d)", _database.lastErrorMessage, (int)result);
}

- (void)bindInt64:(int64_t)n atParameterIndex:(NSUInteger)index
{
    dispatch_assert_queue(_database.queue);
    ASSERT(self._isValid);
    ASSERT_ARG(index, index > 0);

    _WKSQLiteErrorCode result = sqlite3_bind_int64(_handle, index, n);
    if (result != SQLITE_OK)
        RELEASE_LOG_DEBUG(Extensions, "Could not bind integer: %@ (%d)", _database.lastErrorMessage, (int)result);
}

- (void)bindDouble:(double)n atParameterIndex:(NSUInteger)index
{
    dispatch_assert_queue(_database.queue);
    ASSERT(self._isValid);
    ASSERT_ARG(index, index > 0);

    _WKSQLiteErrorCode result = sqlite3_bind_double(_handle, index, n);
    if (result != SQLITE_OK)
        RELEASE_LOG_DEBUG(Extensions, "Could not bind double: %@ (%d)", _database.lastErrorMessage, (int)result);
}

- (void)bindData:(NSData *)data atParameterIndex:(NSUInteger)index
{
    dispatch_assert_queue(_database.queue);
    ASSERT(self._isValid);
    ASSERT_ARG(index, index > 0);

    _WKSQLiteErrorCode result = sqlite3_bind_blob(_handle, index, data.bytes, data.length, SQLITE_TRANSIENT);
    if (result != SQLITE_OK)
        RELEASE_LOG_DEBUG(Extensions, "Could not bind blob: %@ (%d)", _database.lastErrorMessage, (int)result);
}

- (void)bindNullAtParameterIndex:(NSUInteger)index
{
    dispatch_assert_queue(_database.queue);
    ASSERT(self._isValid);
    ASSERT_ARG(index, index > 0);

    _WKSQLiteErrorCode result = sqlite3_bind_null(_handle, index);
    if (result != SQLITE_OK)
        RELEASE_LOG_DEBUG(Extensions, "Could not bind null: %@ (%d)", _database.lastErrorMessage, (int)result);
}

- (NSDictionary *)columnNamesToIndexes
{
    dispatch_assert_queue(_database.queue);
    ASSERT(self._isValid);
    if (_columnNamesToIndexes)
        return _columnNamesToIndexes;

    int columnCount = sqlite3_column_count(_handle);
    NSMutableDictionary *columnNamesToIndexes = [[NSMutableDictionary alloc] initWithCapacity:columnCount];
    for (int i = 0; i < columnCount; ++i) {
        const char* columnName = sqlite3_column_name(_handle, i);
        ASSERT(columnName);

        columnNamesToIndexes[@(columnName)] = @(i);
    }

    _columnNamesToIndexes = columnNamesToIndexes;
    return _columnNamesToIndexes;
}

- (NSArray *)columnNames
{
    dispatch_assert_queue(_database.queue);
    ASSERT(self._isValid);
    if (_columnNames)
        return _columnNames;

    int columnCount = sqlite3_column_count(_handle);
    NSMutableArray *columnNames = [[NSMutableArray alloc] initWithCapacity:columnCount];
    for (int i = 0; i < columnCount; ++i) {
        const char* columnName = sqlite3_column_name(_handle, i);
        [columnNames addObject:@(columnName)];
    }

    _columnNames = columnNames;
    return _columnNames;
}

- (BOOL)_isValid
{
    return !!_handle;
}

@end

#endif // ENABLE(WK_WEB_EXTENSIONS)
