/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 22, 2024.
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
#include "WebExtensionSQLiteDatabase.h"
#include <sqlite3.h>
#include <wtf/Forward.h>
#include <wtf/Noncopyable.h>
#include <wtf/RefPtr.h>
#include <wtf/URL.h>
#include <wtf/WorkQueue.h>

namespace WebKit {

class WebExtensionSQLiteRow;
class WebExtensionSQLiteRowEnumerator;

class WebExtensionSQLiteStatement : public RefCounted<WebExtensionSQLiteStatement> {
    WTF_MAKE_NONCOPYABLE(WebExtensionSQLiteStatement);
    WTF_MAKE_TZONE_ALLOCATED(WebExtensionSQLiteStatement);

public:
    template<typename... Args>
    static Ref<WebExtensionSQLiteStatement> create(Args&&... args)
    {
        return adoptRef(*new WebExtensionSQLiteStatement(std::forward<Args>(args)...));
    }

    explicit WebExtensionSQLiteStatement(Ref<WebExtensionSQLiteDatabase>, const String& query, RefPtr<API::Error>&);

    ~WebExtensionSQLiteStatement();

    void bind(const String&, int parameterIndex);
    void bind(const int&, int parameterIndex);
    void bind(const int64_t&, int parameterIndex);
    void bind(const double&, int parameterIndex);
    void bind(const RefPtr<API::Data>&, int parameterIndex);
    void bind(int parameterIndex);

    int execute();
    bool execute(RefPtr<API::Error>&);

    Ref<WebExtensionSQLiteRowEnumerator> fetch();
    bool fetchWithEnumerationCallback(Function<void(RefPtr<WebExtensionSQLiteRow>, bool)>&, RefPtr<API::Error>&);

    void reset();
    void invalidate();

    Ref<WebExtensionSQLiteDatabase> database() { return m_db; };
    sqlite3_stmt* handle() { return m_handle; };
    bool isValid() { return !!m_handle; };

    Vector<String> columnNames();
    HashMap<String, int> columnNamesToIndicies();
private:
    sqlite3_stmt* m_handle;
    Ref<WebExtensionSQLiteDatabase> m_db;

    Vector<String> m_columnNames;
    HashMap<String, int> m_columnNamesToIndicies;
};

}; // namespace WebKit
