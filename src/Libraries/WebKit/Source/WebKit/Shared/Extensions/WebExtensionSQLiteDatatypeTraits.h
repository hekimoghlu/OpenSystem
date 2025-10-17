/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 7, 2024.
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

#include "APIData.h"
#include "WebExtensionSQLiteDatabase.h"
#include <sqlite3.h>
#include <tuple>

namespace WebKit {

template<typename Type>
class WebExtensionSQLiteDatatypeTraits {
public:
    static Type fetch(sqlite3_stmt* statement, int index);
    static int bind(sqlite3_stmt* statement, int index, const Type&);
};

template<>
class WebExtensionSQLiteDatatypeTraits<int> {
public:
    static inline int fetch(sqlite3_stmt* statement, int index)
    {
        return sqlite3_column_int(statement, index);
    }

    static inline int bind(sqlite3_stmt* statement, int index, int value)
    {
        return sqlite3_bind_int(statement, index, value);
    }
};

template<>
class WebExtensionSQLiteDatatypeTraits<int64_t> {
public:
    static inline int64_t fetch(sqlite3_stmt* statement, int index)
    {
        return sqlite3_column_int64(statement, index);
    }

    static inline int bind(sqlite3_stmt* statement, int index, int64_t value)
    {
        return sqlite3_bind_int64(statement, index, value);
    }
};

template<>
class WebExtensionSQLiteDatatypeTraits<double> {
public:
    static inline double fetch(sqlite3_stmt* statement, int index)
    {
        return sqlite3_column_double(statement, index);
    }

    static inline int bind(sqlite3_stmt* statement, int index, double value)
    {
        return sqlite3_bind_double(statement, index, value);
    }
};

template<>
class WebExtensionSQLiteDatatypeTraits<String> {
public:
    static String fetch(sqlite3_stmt* statement, int index)
    {
        if (sqlite3_column_type(statement, index) == SQLITE_NULL)
            return emptyString();

        return String::fromUTF8(reinterpret_cast<const char*>(sqlite3_column_text(statement, index)));
    }

    static inline int bind(sqlite3_stmt* statement, int index, const String& value)
    {
        if (!value)
            return sqlite3_bind_null(statement, index);

        return sqlite3_bind_text(statement, index, value.utf8().data(), -1, SQLITE_TRANSIENT);
    }
};

template<>
class WebExtensionSQLiteDatatypeTraits<RefPtr<API::Data>> {
public:
    static RefPtr<API::Data> fetch(sqlite3_stmt* statement, int index)
    {
        if (sqlite3_column_type(statement, index) == SQLITE_NULL)
            return nullptr;

        auto* blob = static_cast<const uint8_t*>(sqlite3_column_blob(statement, index));
        if (!blob)
            return nullptr;

        int blobSize = sqlite3_column_bytes(statement, index);
        if (blobSize <= 0)
            return nullptr;

        return API::Data::create(unsafeMakeSpan(blob, blobSize));
    }

    static inline int bind(sqlite3_stmt* statement, int index, RefPtr<API::Data> value)
    {
        if (!value)
            return sqlite3_bind_null(statement, index);

        return sqlite3_bind_blob64(statement, index, value->span().data(), value->span().size(), SQLITE_TRANSIENT);
    }
};

template<>
class WebExtensionSQLiteDatatypeTraits<std::nullptr_t> {
public:
    static inline std::nullptr_t fetch(sqlite3_stmt* statement, int index)
    {
        return std::nullptr_t();
    }

    static inline int bind(sqlite3_stmt* statement, int index, std::nullptr_t)
    {
        return sqlite3_bind_null(statement, index);
    }
};

template<>
class WebExtensionSQLiteDatatypeTraits<decltype(std::ignore)> {
public:
    static inline decltype(std::ignore) fetch(sqlite3_stmt* statement, int index)
    {
        return std::ignore;
    }

    static inline int bind(sqlite3_stmt* statement, int index, decltype(std::ignore))
    {
        return SQLITE_OK;
    }
};

} // namespace WebKit
