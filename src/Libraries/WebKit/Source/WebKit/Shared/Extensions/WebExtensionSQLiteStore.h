/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 8, 2025.
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

#include "WebExtensionSQLiteDatabase.h"
#include <wtf/CompletionHandler.h>
#include <wtf/Forward.h>
#include <wtf/Function.h>
#include <wtf/Markable.h>
#include <wtf/Noncopyable.h>
#include <wtf/UUID.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/WTFString.h>

namespace WebKit {

typedef int SchemaVersion;
typedef int DatabaseResult;
using WTF::UUID;

class WebExtensionSQLiteStore : public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<WebExtensionSQLiteStore> {
    WTF_MAKE_TZONE_ALLOCATED(WebExtensionSQLiteStore);
    WTF_MAKE_NONCOPYABLE(WebExtensionSQLiteStore)

public:
    WebExtensionSQLiteStore(const String& uniqueIdentifier, const String& directory, bool useInMemoryDatabase);
    virtual ~WebExtensionSQLiteStore() { close(); };

    SchemaVersion databaseSchemaVersion();
    bool useInMemoryDatabase() { return m_useInMemoryDatabase; };

    bool openDatabaseIfNecessary(String& outErrorMessage, bool createIfNecessary);

    void close();
    void deleteDatabase(CompletionHandler<void(const String& errorMessage)>&&);
    String deleteDatabaseIfEmpty();

    void createSavepoint(CompletionHandler<void(Markable<WTF::UUID> savepointIdentifier, const String& errorMessage)>&&);
    void commitSavepoint(WTF::UUID& savepointIdentifier, CompletionHandler<void(const String& errorMessage)>&&);
    void rollbackToSavepoint(WTF::UUID& savepointIdentifier, CompletionHandler<void(const String& errorMessage)>&&);

protected:
    virtual DatabaseResult createFreshDatabaseSchema() = 0;
    virtual DatabaseResult resetDatabaseSchema() = 0;
    virtual bool isDatabaseEmpty() = 0;
    virtual SchemaVersion currentDatabaseSchemaVersion() = 0;
    virtual URL databaseURL() = 0;

    DatabaseResult setDatabaseSchemaVersion(SchemaVersion newVersion);
    SchemaVersion migrateToCurrentSchemaVersionIfNeeded();

    Ref<WorkQueue> queue() { return m_queue; };
    RefPtr<WebExtensionSQLiteDatabase> database() { return m_database; };
    String uniqueIdentifier() { return m_uniqueIdentifier; };
    CString lastErrorMessage() { return m_database->m_lastErrorMessage; };
    URL directory() { return m_directory; };

private:
    void vacuum();
    bool isDatabaseOpen();
    String openDatabase(const URL& databaseURL, WebExtensionSQLiteDatabase::AccessType, bool deleteDatabaseFileOnError);
    String deleteDatabaseFileAtURL(const URL& databaseURL, bool reopenDatabase);
    String deleteDatabase();

    String savepointNameFromUUID(const WTF::UUID& savepointIdentifier);

    String handleSchemaVersioning(bool deleteDatabaseFileOnError);

    String m_uniqueIdentifier;
    URL m_directory;
    RefPtr<WebExtensionSQLiteDatabase> m_database;
    Ref<WorkQueue> m_queue;
    bool m_useInMemoryDatabase;
};

} // namespace WebKit
