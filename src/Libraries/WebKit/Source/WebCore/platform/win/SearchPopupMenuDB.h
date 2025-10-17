/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 3, 2023.
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

#include "SQLiteDatabase.h"
#include "SQLiteStatement.h"
#include "SearchPopupMenu.h"
#include <wtf/Noncopyable.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class SearchPopupMenuDB {
    WTF_MAKE_NONCOPYABLE(SearchPopupMenuDB);

public:
    static WEBCORE_EXPORT SearchPopupMenuDB& singleton();
    WEBCORE_EXPORT void saveRecentSearches(const String& name, const Vector<RecentSearch>&);
    WEBCORE_EXPORT void loadRecentSearches(const String& name, Vector<RecentSearch>&);

private:
    SearchPopupMenuDB();
    ~SearchPopupMenuDB();

    bool openDatabase();
    void closeDatabase();
    bool checkDatabaseValidity();
    void deleteAllDatabaseFiles();
    void verifySchemaVersion();
    int executeSQLStatement(Expected<SQLiteStatement, int>&&);
    void checkSQLiteReturnCode(int actual);
    std::unique_ptr<SQLiteStatement> createPreparedStatement(ASCIILiteral sql);

    String m_databaseFilename;
    SQLiteDatabase m_database;
    std::unique_ptr<SQLiteStatement> m_loadSearchTermsForNameStatement;
    std::unique_ptr<SQLiteStatement> m_insertSearchTermStatement;
    std::unique_ptr<SQLiteStatement> m_removeSearchTermsForNameStatement;
};

} // namespace WebCore
