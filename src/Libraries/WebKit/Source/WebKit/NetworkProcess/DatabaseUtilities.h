/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 5, 2022.
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

#include <WebCore/SQLiteDatabase.h>
#include <WebCore/SQLiteTransaction.h>
#include <wtf/RobinHoodHashMap.h>
#include <wtf/Scope.h>

namespace WebCore {
class PrivateClickMeasurement;
class SQLiteStatement;
class SQLiteStatementAutoResetScope;
}

namespace WebKit {

enum class PrivateClickMeasurementAttributionType : bool;

using TableAndIndexPair = std::pair<String, std::optional<String>>;

class DatabaseUtilities {
protected:
    DatabaseUtilities(String&& storageFilePath);
    ~DatabaseUtilities();

    WebCore::SQLiteStatementAutoResetScope scopedStatement(std::unique_ptr<WebCore::SQLiteStatement>&, ASCIILiteral query, ASCIILiteral logString) const;
    ScopeExit<Function<void()>> WARN_UNUSED_RETURN beginTransactionIfNecessary();
    enum class CreatedNewFile : bool { No, Yes };
    CreatedNewFile openDatabaseAndCreateSchemaIfNecessary();
    void enableForeignKeys();
    void close();
    void interrupt();
    virtual bool createSchema() = 0;
    virtual bool createUniqueIndices() = 0;
    virtual void destroyStatements() = 0;
    virtual String getDomainStringFromDomainID(unsigned) const = 0;
    virtual bool needsUpdatedSchema() = 0;
    virtual const MemoryCompactLookupOnlyRobinHoodHashMap<String, TableAndIndexPair>& expectedTableAndIndexQueries() = 0;
    virtual std::span<const ASCIILiteral> sortedTables() = 0;
    TableAndIndexPair currentTableAndIndexQueries(const String&);
    String stripIndexQueryToMatchStoredValue(const char* originalQuery);
    void migrateDataToNewTablesIfNecessary();
    Vector<String> columnsForTable(ASCIILiteral tableName);
    bool addMissingColumnToTable(ASCIILiteral tableName, ASCIILiteral columnName);

    WebCore::PrivateClickMeasurement buildPrivateClickMeasurementFromDatabase(WebCore::SQLiteStatement&, PrivateClickMeasurementAttributionType) const;

    const String m_storageFilePath;
    mutable WebCore::SQLiteDatabase m_database;
    mutable WebCore::SQLiteTransaction m_transaction;
};

} // namespace WebKit
