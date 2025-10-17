/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 28, 2025.
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
#import "_WKWebExtensionSQLiteRow.h"

#if ENABLE(WK_WEB_EXTENSIONS)

#import "_WKWebExtensionSQLiteDatabase.h"
#import "_WKWebExtensionSQLiteStatement.h"
#import <sqlite3.h>

@interface _WKWebExtensionSQLiteRow ()
- (instancetype)init NS_DESIGNATED_INITIALIZER;
@end

@implementation _WKWebExtensionSQLiteRow {
    sqlite3_stmt *_handle;
    _WKWebExtensionSQLiteStatement *_statement;
}

- (instancetype)init
{
    ASSERT_NOT_REACHED();
    return nil;
}

- (instancetype)initWithStatement:(_WKWebExtensionSQLiteStatement *)statement
{
    if (!(self = [super init]))
        return nil;

    _handle = statement.handle;

    _statement = statement;
    dispatch_assert_queue(_statement.database.queue);

    return self;
}

- (instancetype)initWithCurrentRowOfEnumerator:(_WKWebExtensionSQLiteRowEnumerator *)rowEnumerator
{
    return [self initWithStatement:rowEnumerator.statement];
}

#pragma mark - Columns by index

- (NSString *)stringAtIndex:(NSUInteger)index
{
    dispatch_assert_queue(_statement.database.queue);
    if ([self _isNullAtIndex:index])
        return nil;

    sqlite3_stmt* handle = _handle;

    const unsigned char* text = sqlite3_column_text(handle, index);
    int length = sqlite3_column_bytes(handle, index);
    return CFBridgingRelease(CFStringCreateWithBytes(kCFAllocatorDefault, text, length, kCFStringEncodingUTF8, false));
}

- (int)intAtIndex:(NSUInteger)index
{
    dispatch_assert_queue(_statement.database.queue);
    return sqlite3_column_int(_handle, index);
}

- (int64_t)int64AtIndex:(NSUInteger)index
{
    dispatch_assert_queue(_statement.database.queue);
    return sqlite3_column_int64(_handle, index);
}

- (double)doubleAtIndex:(NSUInteger)index
{
    dispatch_assert_queue(_statement.database.queue);
    return sqlite3_column_double(_handle, index);
}

- (BOOL)boolAtIndex:(NSUInteger)index
{
    return !![self intAtIndex:index];
}

- (NSData *)dataAtIndex:(NSUInteger)index
{
    RawData data = [self uncopiedRawDataAtIndex:index];
    if (data.isNull)
        return nil;
    return [NSData dataWithBytes:data.bytes length:data.length];
}

- (NSData *)uncopiedDataAtIndex:(NSUInteger)index
{
    RawData data = [self uncopiedRawDataAtIndex:index];
    if (data.isNull)
        return nil;
    return [NSData dataWithBytesNoCopy:const_cast<void *>(data.bytes) length:data.length freeWhenDone:NO];
}

- (RawData)uncopiedRawDataAtIndex:(NSUInteger)index
{
    dispatch_assert_queue(_statement.database.queue);
    if ([self _isNullAtIndex:index])
        return (RawData) { true, nullptr, 0 };

    sqlite3_stmt* handle = _handle;

    const void* blob = sqlite3_column_blob(handle, index);
    if (!blob)
        return (RawData) { false, nullptr, 0 };

    int length = sqlite3_column_bytes(handle, index);
    return (RawData) { false, blob, length };
}

- (BOOL)_isNullAtIndex:(NSUInteger)index
{
    dispatch_assert_queue(_statement.database.queue);
    return sqlite3_column_type(_handle, index) == SQLITE_NULL;
}

- (NSObject *)objectAtIndex:(NSUInteger)index
{
    switch (sqlite3_column_type(_handle, index)) {
    case SQLITE_INTEGER:
        return @(sqlite3_column_int64(_handle, index));
    case SQLITE_FLOAT:
        return @(sqlite3_column_double(_handle, index));
    case SQLITE_TEXT:
        return [self stringAtIndex:index];
    case SQLITE_NULL:
        return [NSNull null];
    case SQLITE_BLOB:
        return [self dataAtIndex:index];
    default:
        ASSERT_NOT_REACHED();
        return nil;
    }
}

@end

@implementation _WKWebExtensionSQLiteRowEnumerator {
    _WKWebExtensionSQLiteStatement *_statement;
    _WKWebExtensionSQLiteRow *_row;
}

- (instancetype)initWithResultsOfStatement:(_WKWebExtensionSQLiteStatement *)statement
{
    if (!(self = [super init]))
        return nil;

    dispatch_assert_queue(statement.database.queue);
    _statement = statement;

    return self;
}

- (id)nextObject
{
    dispatch_assert_queue(_statement.database.queue);

    _lastResultCode = sqlite3_step(_statement.handle);

    switch (_lastResultCode) {
    case SQLITE_ROW:
        if (!_row)
            _row = [[_WKWebExtensionSQLiteRow alloc] initWithCurrentRowOfEnumerator:self];
        return _row;
    case SQLITE_OK:
    case SQLITE_DONE:
        break;

    default:
        [_statement.database reportErrorWithCode:_lastResultCode statement:_statement.handle error:nullptr];
        break;
    }

    return nil;
}

@end

#endif // ENABLE(WK_WEB_EXTENSIONS)
