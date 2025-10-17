/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 15, 2024.
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

#include "ServiceWorkerTypes.h"
#include "ServiceWorkerUpdateViaCache.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class ServiceWorkerRegistrationKey;
class SQLiteDatabase;
class SQLiteStatement;
class SQLiteStatementAutoResetScope;
class SWScriptStorage;
struct ServiceWorkerContextData;

class SWRegistrationDatabase {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(SWRegistrationDatabase, WEBCORE_EXPORT);
public:
    static constexpr uint64_t schemaVersion = 8;

    WEBCORE_EXPORT SWRegistrationDatabase(const String& path);
    WEBCORE_EXPORT ~SWRegistrationDatabase();
    
    WEBCORE_EXPORT std::optional<Vector<ServiceWorkerContextData>> importRegistrations();
    WEBCORE_EXPORT std::optional<Vector<ServiceWorkerScripts>> updateRegistrations(const Vector<ServiceWorkerContextData>&, const Vector<ServiceWorkerRegistrationKey>&);
    WEBCORE_EXPORT void deleteAllFiles();
    
private:
    void close();
    SWScriptStorage& scriptStorage();
    enum class StatementType : uint8_t {
        GetAllRecords,
        CountAllRecords,
        InsertRecord,
        DeleteRecord,
        Invalid
    };
    ASCIILiteral statementString(StatementType) const;
    SQLiteStatementAutoResetScope cachedStatement(StatementType);
    enum class ShouldCreateIfNotExists : bool { No, Yes };
    bool prepareDatabase(ShouldCreateIfNotExists);
    bool ensureValidRecordsTable();
    std::optional<uint64_t> recordsCount();
    std::optional<Vector<ServiceWorkerContextData>> importRegistrationsImpl();
    std::optional<Vector<ServiceWorkerScripts>> updateRegistrationsImpl(const Vector<ServiceWorkerContextData>&, const Vector<ServiceWorkerRegistrationKey>&);

    String m_directory;
    std::unique_ptr<SQLiteDatabase> m_database;
    Vector<std::unique_ptr<SQLiteStatement>> m_cachedStatements;
    std::unique_ptr<SWScriptStorage> m_scriptStorage;
    
}; // namespace WebCore

struct ImportedScriptAttributes {
    URL responseURL;
    String mimeType;
};

}
