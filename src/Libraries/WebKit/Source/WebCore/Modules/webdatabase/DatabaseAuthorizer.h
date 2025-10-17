/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 20, 2024.
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

#include <wtf/Forward.h>
#include <wtf/HashSet.h>
#include <wtf/ThreadSafeRefCounted.h>
#include <wtf/text/StringHash.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

extern const int SQLAuthAllow;
extern const int SQLAuthIgnore;
extern const int SQLAuthDeny;

class DatabaseAuthorizer : public ThreadSafeRefCounted<DatabaseAuthorizer> {
public:

    enum Permissions {
        ReadWriteMask = 0,
        ReadOnlyMask = 1 << 1,
        NoAccessMask = 1 << 2
    };

    static Ref<DatabaseAuthorizer> create(const String& databaseInfoTableName);

    int createTable(const String& tableName);
    int createTempTable(const String& tableName);
    int dropTable(const String& tableName);
    int dropTempTable(const String& tableName);
    int allowAlterTable(const String& databaseName, const String& tableName);

    int createIndex(const String& indexName, const String& tableName);
    int createTempIndex(const String& indexName, const String& tableName);
    int dropIndex(const String& indexName, const String& tableName);
    int dropTempIndex(const String& indexName, const String& tableName);

    int createTrigger(const String& triggerName, const String& tableName);
    int createTempTrigger(const String& triggerName, const String& tableName);
    int dropTrigger(const String& triggerName, const String& tableName);
    int dropTempTrigger(const String& triggerName, const String& tableName);

    int createView(const String& viewName);
    int createTempView(const String& viewName);
    int dropView(const String& viewName);
    int dropTempView(const String& viewName);

    int createVTable(const String& tableName, const String& moduleName);
    int dropVTable(const String& tableName, const String& moduleName);

    int allowDelete(const String& tableName);
    int allowInsert(const String& tableName);
    int allowUpdate(const String& tableName, const String& columnName);
    int allowTransaction();

    int allowSelect() { return SQLAuthAllow; }
    int allowRead(const String& tableName, const String& columnName);

    int allowReindex(const String& indexName);
    int allowAnalyze(const String& tableName);
    int allowFunction(const String& functionName);
    int allowPragma(const String& pragmaName, const String& firstArgument);

    int allowAttach(const String& filename);
    int allowDetach(const String& databaseName);

    void disable();
    void enable();
    void setPermissions(int permissions);

    void reset();
    void resetDeletes();

    bool lastActionWasInsert() const { return m_lastActionWasInsert; }
    bool lastActionChangedDatabase() const { return m_lastActionChangedDatabase; }
    bool hadDeletes() const { return m_hadDeletes; }

private:
    explicit DatabaseAuthorizer(const String& databaseInfoTableName);
    void addAllowedFunctions();
    int denyBasedOnTableName(const String&) const;
    int updateDeletesBasedOnTableName(const String&);
    bool allowWrite();

    int m_permissions;
    bool m_securityEnabled : 1;
    bool m_lastActionWasInsert : 1;
    bool m_lastActionChangedDatabase : 1;
    bool m_hadDeletes : 1;

    const String m_databaseInfoTableName;

    HashSet<String, ASCIICaseInsensitiveHash> m_allowedFunctions;
};

} // namespace WebCore
