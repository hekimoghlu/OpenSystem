/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 17, 2024.
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
#import "_WKWebExtensionSQLiteDatabase.h"
#import <sqlite3.h>
#import <tuple>

NS_ASSUME_NONNULL_BEGIN

namespace WebKit {

template<typename Type>
class _WKWebExtensionSQLiteDatatypeTraits {
public:
    static Type _Nullable fetch(sqlite3_stmt* statement, int index);
    static _WKSQLiteErrorCode bind(sqlite3_stmt* statement, int index, const Type _Nullable &);
};

template<>
class _WKWebExtensionSQLiteDatatypeTraits<int> {
public:
    static inline int fetch(sqlite3_stmt* statement, int index)
    {
        return sqlite3_column_int(statement, index);
    }

    static inline _WKSQLiteErrorCode bind(sqlite3_stmt* statement, int index, int value)
    {
        return sqlite3_bind_int(statement, index, value);
    }
};

template<>
class _WKWebExtensionSQLiteDatatypeTraits<int64_t> {
public:
    static inline int64_t fetch(sqlite3_stmt* statement, int index)
    {
        return sqlite3_column_int64(statement, index);
    }

    static inline _WKSQLiteErrorCode bind(sqlite3_stmt* statement, int index, int64_t value)
    {
        return sqlite3_bind_int64(statement, index, value);
    }
};

template<>
class _WKWebExtensionSQLiteDatatypeTraits<double> {
public:
    static inline double fetch(sqlite3_stmt* statement, int index)
    {
        return sqlite3_column_double(statement, index);
    }

    static inline _WKSQLiteErrorCode bind(sqlite3_stmt* statement, int index, double value)
    {
        return sqlite3_bind_double(statement, index, value);
    }
};

template<>
class _WKWebExtensionSQLiteDatatypeTraits<NSString *> {
public:
    static NSString * _Nullable fetch(sqlite3_stmt* statement, int index)
    {
        if (sqlite3_column_type(statement, index) == SQLITE_NULL)
            return nil;

        return CFBridgingRelease(CFStringCreateWithBytes(kCFAllocatorDefault, sqlite3_column_text(statement, index), sqlite3_column_bytes(statement, index), kCFStringEncodingUTF8, false));
    }

    static inline _WKSQLiteErrorCode bind(sqlite3_stmt* statement, int index, NSString * _Nullable value)
    {
        if (!value)
            return sqlite3_bind_null(statement, index);

        return sqlite3_bind_text(statement, index, value.UTF8String, -1, SQLITE_TRANSIENT);
    }
};

template<>
class _WKWebExtensionSQLiteDatatypeTraits<NSData *> {
public:
    static NSData * _Nullable fetch(sqlite3_stmt* statement, int index)
    {
        if (sqlite3_column_type(statement, index) == SQLITE_NULL)
            return nil;

        return CFBridgingRelease(CFDataCreate(kCFAllocatorDefault, (const UInt8 *)sqlite3_column_blob(statement, index), sqlite3_column_bytes(statement, index)));
    }

    static inline _WKSQLiteErrorCode bind(sqlite3_stmt* statement, int index, NSData * _Nullable value)
    {
        if (!value)
            return sqlite3_bind_null(statement, index);

        return sqlite3_bind_blob64(statement, index, value.bytes, value.length, SQLITE_TRANSIENT);
    }
};

template<>
class _WKWebExtensionSQLiteDatatypeTraits<NSDate *> {
public:
    static inline NSDate * _Nullable fetch(sqlite3_stmt* statement, int index)
    {
        if (sqlite3_column_type(statement, index) == SQLITE_NULL)
            return nil;

        return [NSDate dateWithTimeIntervalSinceReferenceDate:sqlite3_column_double(statement, index)];
    }

    static inline _WKSQLiteErrorCode bind(sqlite3_stmt* statement, int index, NSDate * _Nullable value)
    {
        if (!value)
            return sqlite3_bind_null(statement, index);

        return sqlite3_bind_double(statement, index, value.timeIntervalSinceReferenceDate);
    }
};

template<>
class _WKWebExtensionSQLiteDatatypeTraits<NSNumber *> {
public:
    static NSNumber * _Nullable fetch(sqlite3_stmt* statement, int index)
    {
        switch (sqlite3_column_type(statement, index)) {
        case SQLITE_INTEGER:
            return @(sqlite3_column_int64(statement, index));
        default:
        case SQLITE_FLOAT:
            return @(sqlite3_column_double(statement, index));
        case SQLITE_NULL:
            return nil;
        }
    }

    static _WKSQLiteErrorCode bind(sqlite3_stmt* statement, int index, NSNumber * _Nullable value)
    {
        if (!value)
            return sqlite3_bind_null(statement, index);

        const char* objCType = [value objCType];

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
        if (!strcmp(objCType, @encode(double)) || !strcmp(objCType, @encode(float)))
            return sqlite3_bind_double(statement, index, value.doubleValue);
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

        return sqlite3_bind_int64(statement, index, value.longLongValue);
    }
};

template<>
class _WKWebExtensionSQLiteDatatypeTraits<NSObject *> {
public:
    static NSObject * _Nullable fetch(sqlite3_stmt* statement, int index)
    {
        switch (sqlite3_column_type(statement, index)) {
        case SQLITE_INTEGER:
            return @(sqlite3_column_int64(statement, index));
        case SQLITE_FLOAT:
            return @(sqlite3_column_double(statement, index));
        case SQLITE_NULL:
            return nil;
        case SQLITE_BLOB:
            return _WKWebExtensionSQLiteDatatypeTraits<NSData *>::fetch(statement, index);
        case SQLITE_TEXT:
            return _WKWebExtensionSQLiteDatatypeTraits<NSString *>::fetch(statement, index);
        default:
            ASSERT_NOT_REACHED();
            return nil;
        }
    }

    static _WKSQLiteErrorCode bind(sqlite3_stmt* statement, int index, NSObject * _Nullable value)
    {
        if (!value)
            return sqlite3_bind_null(statement, index);

        if ([value isKindOfClass:[NSString class]])
            return sqlite3_bind_text(statement, index, ((NSString *)value).UTF8String, -1, SQLITE_TRANSIENT);

        if ([value isKindOfClass:[NSData class]])
            return sqlite3_bind_blob(statement, index, ((NSData *)value).bytes, (int)((NSData *)value).length, SQLITE_TRANSIENT);

        if ([value isKindOfClass:[NSNumber class]])
            return _WKWebExtensionSQLiteDatatypeTraits<NSNumber *>::bind(statement, index, (NSNumber *)value);

        if ([value isKindOfClass:[NSDate class]])
            return sqlite3_bind_double(statement, index, ((NSDate *)value).timeIntervalSinceReferenceDate);

        ASSERT_NOT_REACHED();
        return SQLITE_ERROR;
    }
};

template<>
class _WKWebExtensionSQLiteDatatypeTraits<std::nullptr_t> {
public:
    static inline std::nullptr_t fetch(sqlite3_stmt* statement, int index)
    {
        return std::nullptr_t();
    }

    static inline _WKSQLiteErrorCode bind(sqlite3_stmt* statement, int index, std::nullptr_t)
    {
        return sqlite3_bind_null(statement, index);
    }
};

template<>
class _WKWebExtensionSQLiteDatatypeTraits<decltype(std::ignore)> {
public:
    static inline decltype(std::ignore) fetch(sqlite3_stmt* statement, int index)
    {
        return std::ignore;
    }

    static inline _WKSQLiteErrorCode bind(sqlite3_stmt* statement, int index, decltype(std::ignore))
    {
        return SQLITE_OK;
    }
};

} // namespace WebKit

NS_ASSUME_NONNULL_END
