/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 3, 2024.
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

#include "ExceptionOr.h"
#include "SQLiteDatabase.h"
#include <wtf/Deque.h>
#include <wtf/Lock.h>

namespace WebCore {

class DatabaseCallback;
class DatabaseContext;
class DatabaseDetails;
class DatabaseThread;
class Document;
class SecurityOrigin;
class SQLTransaction;
class SQLTransactionBackend;
class SQLTransactionCallback;
class SQLTransactionCoordinator;
class SQLTransactionErrorCallback;
class SQLTransactionWrapper;
class VoidCallback;
class SecurityOriginData;

using DatabaseGUID = int;

class Database : public ThreadSafeRefCounted<Database> {
public:
    ~Database();

    ExceptionOr<void> openAndVerifyVersion(bool setVersionInNewDatabase);
    void close();

    void interrupt();

    bool opened() const { return m_opened; }
    bool isNew() const { return m_new; }

    unsigned long long maximumSize();

    void scheduleTransactionStep(SQLTransaction&);
    void inProgressTransactionCompleted();

    bool hasPendingTransaction();

    bool hasPendingCreationEvent() const { return m_hasPendingCreationEvent; }
    void setHasPendingCreationEvent(bool value) { m_hasPendingCreationEvent = value; }

    void didCommitWriteTransaction();
    bool didExceedQuota();

    SQLTransactionCoordinator* transactionCoordinator();

    // Direct support for the DOM API
    String version() const;
    void changeVersion(String&& oldVersion, String&& newVersion, RefPtr<SQLTransactionCallback>&&, RefPtr<SQLTransactionErrorCallback>&&, RefPtr<VoidCallback>&& successCallback);
    void transaction(Ref<SQLTransactionCallback>&&, RefPtr<SQLTransactionErrorCallback>&&, RefPtr<VoidCallback>&& successCallback);
    void readTransaction(Ref<SQLTransactionCallback>&&, RefPtr<SQLTransactionErrorCallback>&&, RefPtr<VoidCallback>&& successCallback);

    // Internal engine support
    String stringIdentifierIsolatedCopy() const;
    String displayNameIsolatedCopy() const;
    String expectedVersionIsolatedCopy() const;
    unsigned long long estimatedSize() const;
    String fileNameIsolatedCopy() const;
    DatabaseDetails details() const;
    SQLiteDatabase& sqliteDatabase() { return m_sqliteDatabase; }

    void disableAuthorizer();
    void enableAuthorizer();
    void setAuthorizerPermissions(int);
    bool lastActionChangedDatabase();
    bool lastActionWasInsert();
    void resetDeletes();
    bool hadDeletes();
    void resetAuthorizer();

    DatabaseContext& databaseContext() { return m_databaseContext; }
    DatabaseThread& databaseThread();
    Document& document() { return m_document; }
    void logErrorMessage(const String& message);

    Vector<String> tableNames();

    SecurityOriginData securityOrigin();

    void markAsDeletedAndClose();
    bool deleted() const { return m_deleted; }

    void scheduleTransactionCallback(SQLTransaction*);

    void incrementalVacuumIfNeeded();

    // Called from DatabaseTask
    ExceptionOr<void> performOpenAndVerify(bool shouldSetVersionInNewDatabase);
    Vector<String> performGetTableNames();

    // Called from DatabaseTask and DatabaseThread
    void performClose();

private:
    Database(DatabaseContext&, const String& name, const String& expectedVersion, const String& displayName, unsigned long long estimatedSize);

    void closeDatabase();

    bool getVersionFromDatabase(String& version, bool shouldCacheVersion = true);
    bool setVersionInDatabase(const String& version, bool shouldCacheVersion = true);
    void setExpectedVersion(const String&);
    String getCachedVersion() const;
    void setCachedVersion(const String&);
    bool getActualVersionForTransaction(String& version);
    void setEstimatedSize(unsigned long long);

    void scheduleTransaction() WTF_REQUIRES_LOCK(m_transactionInProgressLock);

    void runTransaction(RefPtr<SQLTransactionCallback>&&, RefPtr<SQLTransactionErrorCallback>&&, RefPtr<VoidCallback>&& successCallback, RefPtr<SQLTransactionWrapper>&&, bool readOnly);

#if !LOG_DISABLED || !ERROR_DISABLED
    String databaseDebugName() const;
#endif

    Ref<Document> m_document;
    Ref<SecurityOrigin> m_contextThreadSecurityOrigin;
    Ref<SecurityOrigin> m_databaseThreadSecurityOrigin;
    Ref<DatabaseContext> m_databaseContext;

    bool m_deleted { false };
    bool m_hasPendingCreationEvent { false };

    String m_name;
    String m_expectedVersion;
    String m_displayName;
    unsigned long long m_estimatedSize;
    String m_filename;

    DatabaseGUID m_guid;
    bool m_opened { false };
    bool m_new { false };

    SQLiteDatabase m_sqliteDatabase;

    Ref<DatabaseAuthorizer> m_databaseAuthorizer;

    Deque<Ref<SQLTransaction>> m_transactionQueue WTF_GUARDED_BY_LOCK(m_transactionInProgressLock);
    Lock m_transactionInProgressLock;
    bool m_transactionInProgress WTF_GUARDED_BY_LOCK(m_transactionInProgressLock) { false };
    bool m_isTransactionQueueEnabled WTF_GUARDED_BY_LOCK(m_transactionInProgressLock) { true };

    friend class ChangeVersionWrapper;
    friend class DatabaseManager;
    friend class SQLTransaction;
    friend class SQLTransactionBackend;
};

} // namespace WebCore
