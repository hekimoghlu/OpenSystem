/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 1, 2022.
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
#include "config.h"
#include "DatabaseManager.h"

#include "Database.h"
#include "DatabaseCallback.h"
#include "DatabaseContext.h"
#include "DatabaseTask.h"
#include "DatabaseTracker.h"
#include "Document.h"
#include "InspectorInstrumentation.h"
#include "Logging.h"
#include "PlatformStrategies.h"
#include "ScriptController.h"
#include "SecurityOrigin.h"
#include "SecurityOriginData.h"
#include "WindowEventLoop.h"
#include <wtf/NeverDestroyed.h>

namespace WebCore {

class DatabaseManager::ProposedDatabase {
public:
    ProposedDatabase(DatabaseManager&, SecurityOrigin&, const String& name, const String& displayName, unsigned long estimatedSize);
    ~ProposedDatabase();

    SecurityOrigin& origin() { return m_origin; }
    DatabaseDetails& details() { return m_details; }

private:
    DatabaseManager& m_manager;
    Ref<SecurityOrigin> m_origin;
    DatabaseDetails m_details;
};

DatabaseManager::ProposedDatabase::ProposedDatabase(DatabaseManager& manager, SecurityOrigin& origin, const String& name, const String& displayName, unsigned long estimatedSize)
    : m_manager(manager)
    , m_origin(origin.isolatedCopy())
    , m_details(name.isolatedCopy(), displayName.isolatedCopy(), estimatedSize, 0, std::nullopt, std::nullopt)
{
    m_manager.addProposedDatabase(*this);
}

inline DatabaseManager::ProposedDatabase::~ProposedDatabase()
{
    m_manager.removeProposedDatabase(*this);
}

DatabaseManager& DatabaseManager::singleton()
{
    static NeverDestroyed<DatabaseManager> instance;
    return instance;
}

void DatabaseManager::initialize(const String& databasePath)
{
    platformInitialize(databasePath);
    DatabaseTracker::initializeTracker(databasePath);
}

void DatabaseManager::setClient(DatabaseManagerClient* client)
{
    m_client = client;
    DatabaseTracker::singleton().setClient(client);
}

bool DatabaseManager::isAvailable()
{
    return m_databaseIsAvailable;
}

void DatabaseManager::setIsAvailable(bool available)
{
    m_databaseIsAvailable = available;
}

Ref<DatabaseContext> DatabaseManager::databaseContext(Document& document)
{
    if (auto databaseContext = document.databaseContext())
        return *databaseContext;
    auto context = adoptRef(*new DatabaseContext(document));
    context->suspendIfNeeded();
    return context;
}

#if LOG_DISABLED

static inline void logOpenDatabaseError(Document&, const String&)
{
}

#else

static void logOpenDatabaseError(Document& document, const String& name)
{
    LOG(StorageAPI, "Database %s for origin %s not allowed to be established", name.utf8().data(), document.securityOrigin().toString().utf8().data());
}

#endif

ExceptionOr<Ref<Database>> DatabaseManager::openDatabaseBackend(Document& document, const String& name, const String& expectedVersion, const String& displayName, unsigned estimatedSize, bool setVersionInNewDatabase)
{
    auto backend = tryToOpenDatabaseBackend(document, name, expectedVersion, displayName, estimatedSize, setVersionInNewDatabase, FirstTryToOpenDatabase);

    if (backend.hasException()) {
        if (backend.exception().code() == ExceptionCode::QuotaExceededError) {
            // Notify the client that we've exceeded the database quota.
            // The client may want to increase the quota, and we'll give it
            // one more try after if that is the case.
            {
                ProposedDatabase proposedDatabase { *this, document.securityOrigin(), name, displayName, estimatedSize };
                this->databaseContext(document)->databaseExceededQuota(name, proposedDatabase.details());
            }
            backend = tryToOpenDatabaseBackend(document, name, expectedVersion, displayName, estimatedSize, setVersionInNewDatabase, RetryOpenDatabase);
        }
    }

    if (backend.hasException()) {
        if (backend.exception().code() == ExceptionCode::InvalidStateError)
            logErrorMessage(document, backend.exception().message());
        else
            logOpenDatabaseError(document, name);
    }

    return backend;
}

ExceptionOr<Ref<Database>> DatabaseManager::tryToOpenDatabaseBackend(Document& document, const String& name, const String& expectedVersion, const String& displayName, unsigned estimatedSize, bool setVersionInNewDatabase,
    OpenAttempt attempt)
{
    auto* page = document.page();
    if (!page || page->usesEphemeralSession())
        return Exception { ExceptionCode::SecurityError };

    auto backendContext = this->databaseContext(document);

    ExceptionOr<void> preflightResult;
    switch (attempt) {
    case FirstTryToOpenDatabase:
        preflightResult = DatabaseTracker::singleton().canEstablishDatabase(backendContext, name, estimatedSize);
        break;
    case RetryOpenDatabase:
        preflightResult = DatabaseTracker::singleton().retryCanEstablishDatabase(backendContext, name, estimatedSize);
        break;
    }
    if (preflightResult.hasException())
        return preflightResult.releaseException();

    auto database = adoptRef(*new Database(backendContext, name, expectedVersion, displayName, estimatedSize));

    auto openResult = database->openAndVerifyVersion(setVersionInNewDatabase);
    if (openResult.hasException())
        return openResult.releaseException();

    // FIXME: What guarantees backendContext.securityOrigin() is non-null?
    DatabaseTracker::singleton().setDatabaseDetails(backendContext->securityOrigin(), name, displayName, estimatedSize);
    return database;
}

void DatabaseManager::addProposedDatabase(ProposedDatabase& database)
{
    Locker locker { m_proposedDatabasesLock };
    m_proposedDatabases.add(&database);
}

void DatabaseManager::removeProposedDatabase(ProposedDatabase& database)
{
    Locker locker { m_proposedDatabasesLock };
    m_proposedDatabases.remove(&database);
}

ExceptionOr<Ref<Database>> DatabaseManager::openDatabase(Document& document, const String& name, const String& expectedVersion, const String& displayName, unsigned estimatedSize, RefPtr<DatabaseCallback>&& creationCallback)
{
    ASSERT(isMainThread());

    // FIXME: Remove this call to ScriptController::initializeMainThread(). The
    // main thread should have been initialized by a WebKit entrypoint already.
    // Also, initializeMainThread() does nothing on iOS.
    ScriptController::initializeMainThread();

    bool setVersionInNewDatabase = !creationCallback;
    auto openResult = openDatabaseBackend(document, name, expectedVersion, displayName, estimatedSize, setVersionInNewDatabase);
    if (openResult.hasException())
        return openResult.releaseException();

    RefPtr<Database> database = openResult.releaseReturnValue();

    auto databaseContext = this->databaseContext(document);
    databaseContext->setHasOpenDatabases();
    InspectorInstrumentation::didOpenDatabase(*database);

    if (database->isNew() && creationCallback.get()) {
        LOG(StorageAPI, "Scheduling DatabaseCreationCallbackTask for database %p\n", database.get());
        database->setHasPendingCreationEvent(true);
        database->m_document->eventLoop().queueTask(TaskSource::Networking, [creationCallback, database]() {
            creationCallback->handleEvent(*database);
            database->setHasPendingCreationEvent(false);
        });
    }

    return database.releaseNonNull();
}

bool DatabaseManager::hasOpenDatabases(Document& document)
{
    auto databaseContext = document.databaseContext();
    return databaseContext && databaseContext->hasOpenDatabases();
}

void DatabaseManager::stopDatabases(Document& document, DatabaseTaskSynchronizer* synchronizer)
{
    auto databaseContext = document.databaseContext();
    if (!databaseContext || !databaseContext->stopDatabases(synchronizer)) {
        if (synchronizer)
            synchronizer->taskCompleted();
    }
}

String DatabaseManager::fullPathForDatabase(SecurityOrigin& origin, const String& name, bool createIfDoesNotExist)
{
    {
        Locker locker { m_proposedDatabasesLock };
        for (auto* proposedDatabase : m_proposedDatabases) {
            if (proposedDatabase->details().name() == name && proposedDatabase->origin().equal(origin))
                return String();
        }
    }
    return DatabaseTracker::singleton().fullPathForDatabase(origin.data(), name, createIfDoesNotExist);
}

DatabaseDetails DatabaseManager::detailsForNameAndOrigin(const String& name, SecurityOrigin& origin)
{
    {
        Locker locker { m_proposedDatabasesLock };
        for (auto* proposedDatabase : m_proposedDatabases) {
            if (proposedDatabase->details().name() == name && proposedDatabase->origin().equal(origin)) {
                ASSERT(&proposedDatabase->details().thread() == &Thread::current() || isMainThread());
                return proposedDatabase->details();
            }
        }
    }

    return DatabaseTracker::singleton().detailsForNameAndOrigin(name, origin.data());
}

void DatabaseManager::logErrorMessage(Document& document, const String& message)
{
    document.addConsoleMessage(MessageSource::Storage, MessageLevel::Error, message);
}

#if !PLATFORM(COCOA)
void DatabaseManager::platformInitialize(const String& databasePath)
{
    UNUSED_PARAM(databasePath);
}
#endif

} // namespace WebCore
