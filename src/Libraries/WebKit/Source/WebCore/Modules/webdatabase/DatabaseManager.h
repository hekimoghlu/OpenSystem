/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 22, 2025.
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

#include "DatabaseDetails.h"
#include "ExceptionOr.h"
#include <wtf/Assertions.h>
#include <wtf/HashSet.h>
#include <wtf/Lock.h>

namespace WebCore {

class Database;
class DatabaseCallback;
class DatabaseContext;
class DatabaseManagerClient;
class DatabaseTaskSynchronizer;
class Document;
class Exception;
class SecurityOrigin;
class SecurityOriginData;

class DatabaseManager {
    WTF_MAKE_NONCOPYABLE(DatabaseManager);
    friend class WTF::NeverDestroyed<DatabaseManager>;
public:
    WEBCORE_EXPORT static DatabaseManager& singleton();

    WEBCORE_EXPORT void initialize(const String& databasePath);
    WEBCORE_EXPORT void setClient(DatabaseManagerClient*);

    bool isAvailable();
    WEBCORE_EXPORT void setIsAvailable(bool);

    // This gets a DatabaseContext for the specified Document.
    // If one doesn't already exist, it will create a new one.
    Ref<DatabaseContext> databaseContext(Document&);

    ExceptionOr<Ref<Database>> openDatabase(Document&, const String& name, const String& expectedVersion, const String& displayName, unsigned estimatedSize, RefPtr<DatabaseCallback>&&);

    WEBCORE_EXPORT bool hasOpenDatabases(Document&);
    void stopDatabases(Document&, DatabaseTaskSynchronizer*);

    WEBCORE_EXPORT String fullPathForDatabase(SecurityOrigin&, const String& name, bool createIfDoesNotExist = true);

    WEBCORE_EXPORT DatabaseDetails detailsForNameAndOrigin(const String&, SecurityOrigin&);

private:
    DatabaseManager() = default;
    ~DatabaseManager() = delete;

    void platformInitialize(const String& databasePath);

    enum OpenAttempt { FirstTryToOpenDatabase, RetryOpenDatabase };
    ExceptionOr<Ref<Database>> openDatabaseBackend(Document&, const String& name, const String& expectedVersion, const String& displayName, unsigned estimatedSize, bool setVersionInNewDatabase);
    ExceptionOr<Ref<Database>> tryToOpenDatabaseBackend(Document&, const String& name, const String& expectedVersion, const String& displayName, unsigned estimatedSize, bool setVersionInNewDatabase, OpenAttempt);

    class ProposedDatabase;
    void addProposedDatabase(ProposedDatabase&);
    void removeProposedDatabase(ProposedDatabase&);

    static void logErrorMessage(Document&, const String& message);

    DatabaseManagerClient* m_client { nullptr };
    bool m_databaseIsAvailable { true };

    Lock m_proposedDatabasesLock;
    HashSet<ProposedDatabase*> m_proposedDatabases WTF_GUARDED_BY_LOCK(m_proposedDatabasesLock);
};

} // namespace WebCore
