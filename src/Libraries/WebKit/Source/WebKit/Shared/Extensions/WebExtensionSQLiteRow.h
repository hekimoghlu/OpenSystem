/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 3, 2023.
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
#include "APIError.h"
#include "WebExtensionSQLiteStatement.h"
#include <sqlite3.h>
#include <wtf/Forward.h>
#include <wtf/Noncopyable.h>
#include <wtf/RefPtr.h>
#include <wtf/URL.h>
#include <wtf/WorkQueue.h>

namespace WebKit {

class WebExtensionSQLiteRow : public RefCounted<WebExtensionSQLiteRow> {
    WTF_MAKE_NONCOPYABLE(WebExtensionSQLiteRow);
    WTF_MAKE_TZONE_ALLOCATED(WebExtensionSQLiteRow);

public:
    template<typename... Args>
    static Ref<WebExtensionSQLiteRow> create(Args&&... args)
    {
        return adoptRef(*new WebExtensionSQLiteRow(std::forward<Args>(args)...));
    }
    explicit WebExtensionSQLiteRow(Ref<WebExtensionSQLiteStatement>);

    String getString(int index);
    int getInt(int index);
    int64_t getInt64(int index);
    double getDouble(int index);
    bool getBool(int index);
    RefPtr<API::Data> getData(int index);

private:
    bool isNullAtIndex(int index);

    Ref<WebExtensionSQLiteStatement> m_statement;
    sqlite3_stmt* m_handle;
};

class WebExtensionSQLiteRowEnumerator : public RefCounted<WebExtensionSQLiteRowEnumerator> {
    WTF_MAKE_NONCOPYABLE(WebExtensionSQLiteRowEnumerator);
    WTF_MAKE_TZONE_ALLOCATED(WebExtensionSQLiteRowEnumerator);

public:
    template<typename... Args>
    static Ref<WebExtensionSQLiteRowEnumerator> create(Args&&... args)
    {
        return adoptRef(*new WebExtensionSQLiteRowEnumerator(std::forward<Args>(args)...));
    }

    explicit WebExtensionSQLiteRowEnumerator(Ref<WebExtensionSQLiteStatement>);

    RefPtr<WebExtensionSQLiteRow> next();
    Ref<WebExtensionSQLiteStatement> statement() { return m_statement; };

private:
    Ref<WebExtensionSQLiteStatement> m_statement;
    RefPtr<WebExtensionSQLiteRow> m_row;
};

}; // namespace WebKit
