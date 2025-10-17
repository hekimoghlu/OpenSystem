/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 23, 2022.
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
#include "LocalStorageManager.h"

#include "MemoryStorageArea.h"
#include "SQLiteStorageArea.h"
#include "StorageAreaRegistry.h"
#include <WebCore/SecurityOriginData.h>
#include <wtf/FileSystem.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/MakeString.h>

namespace WebKit {

// Suggested by https://www.w3.org/TR/webstorage/#disk-space
constexpr unsigned localStorageQuotaInBytes = 5 * MB;
constexpr auto s_fileSuffix = ".localstorage"_s;
constexpr auto s_fileName = "localstorage.sqlite3"_s;

// This is intended to be used for existing files.
// We should not include origin in file name.
static std::optional<WebCore::SecurityOriginData> fileNameToOrigin(const String& fileName)
{
    if (!fileName.endsWith(StringView { s_fileSuffix }))
        return std::nullopt;

    auto suffixLength = s_fileSuffix.length();
    auto fileNameLength = fileName.length();
    if (fileNameLength <= suffixLength)
        return std::nullopt;

    auto originIdentifier = fileName.left(fileNameLength - suffixLength);
    return WebCore::SecurityOriginData::fromDatabaseIdentifier(originIdentifier);
}

static String originToFileName(const WebCore::ClientOrigin& origin)
{
    auto databaseIdentifier = origin.clientOrigin.optionalDatabaseIdentifier();
    if (databaseIdentifier.isEmpty())
        return { };

    return makeString(databaseIdentifier, ".localstorage"_s);
}

WTF_MAKE_TZONE_ALLOCATED_IMPL(LocalStorageManager);

Vector<WebCore::SecurityOriginData> LocalStorageManager::originsOfLocalStorageData(const String& path)
{
    Vector<WebCore::SecurityOriginData> origins;
    if (path.isEmpty())
        return origins;

    for (auto& fileName : FileSystem::listDirectory(path)) {
        if (auto origin = fileNameToOrigin(fileName))
            origins.append(*origin);
    }

    return origins;
}

String LocalStorageManager::localStorageFilePath(const String& directory, const WebCore::ClientOrigin& origin)
{
    if (directory.isEmpty())
        return emptyString();

    auto fileName = originToFileName(origin);
    if (fileName.isEmpty())
        return emptyString();

    return FileSystem::pathByAppendingComponent(directory, fileName);
}

String LocalStorageManager::localStorageFilePath(const String& directory)
{
    if (directory.isEmpty())
        return emptyString();

    return FileSystem::pathByAppendingComponent(directory, s_fileName);
}

LocalStorageManager::LocalStorageManager(const String& path, StorageAreaRegistry& registry)
    : m_path(path)
    , m_registry(registry)
{
}

bool LocalStorageManager::isActive() const
{
    return (m_localStorageArea && m_localStorageArea->hasListeners()) || (m_transientStorageArea && m_transientStorageArea->hasListeners());
}

bool LocalStorageManager::hasDataInMemory() const
{
    RefPtr memoryLocalStorage = dynamicDowncast<MemoryStorageArea>(m_localStorageArea);
    if (memoryLocalStorage && !memoryLocalStorage->isEmpty())
        return true;

    RefPtr transientStorage = m_transientStorageArea;
    return transientStorage && !transientStorage->isEmpty();
}

void LocalStorageManager::clearDataInMemory()
{
    if (RefPtr storage = dynamicDowncast<MemoryStorageArea>(m_localStorageArea))
        storage->clear();

    if (RefPtr transientStorage = m_transientStorageArea)
        transientStorage->clear();
}

void LocalStorageManager::clearDataOnDisk()
{
    if (RefPtr storage = dynamicDowncast<SQLiteStorageArea>(m_localStorageArea))
        storage->clear();
}

void LocalStorageManager::close()
{
    if (RefPtr storage = dynamicDowncast<SQLiteStorageArea>(m_localStorageArea))
        storage->close();
}

void LocalStorageManager::handleLowMemoryWarning()
{
    if (RefPtr storage = dynamicDowncast<SQLiteStorageArea>(m_localStorageArea))
        storage->handleLowMemoryWarning();
}

void LocalStorageManager::syncLocalStorage()
{
    if (RefPtr storage = dynamicDowncast<SQLiteStorageArea>(m_localStorageArea))
        storage->commitTransactionIfNecessary();
}

void LocalStorageManager::connectionClosed(IPC::Connection::UniqueID connection)
{
    connectionClosedForLocalStorageArea(connection);
    connectionClosedForTransientStorageArea(connection);
}

void LocalStorageManager::connectionClosedForLocalStorageArea(IPC::Connection::UniqueID connection)
{
    RefPtr storage = m_localStorageArea;
    if (!storage)
        return;

    storage->removeListener(connection);
    if (storage->hasListeners())
        return;

    if (RefPtr memoryStorage = dynamicDowncast<MemoryStorageArea>(*storage); memoryStorage && !memoryStorage->isEmpty())
        return;

    m_registry->unregisterStorageArea(storage->identifier());
    m_localStorageArea = nullptr;
}

void LocalStorageManager::connectionClosedForTransientStorageArea(IPC::Connection::UniqueID connection)
{
    RefPtr transientStorageArea = m_transientStorageArea;
    if (!transientStorageArea)
        return;

    transientStorageArea->removeListener(connection);
    if (transientStorageArea->hasListeners() || !transientStorageArea->isEmpty())
        return;

    m_registry->unregisterStorageArea(transientStorageArea->identifier());
    m_transientStorageArea = nullptr;
}

StorageAreaBase& LocalStorageManager::ensureLocalStorageArea(const WebCore::ClientOrigin& origin, Ref<WorkQueue>&& workQueue)
{
    if (!m_localStorageArea) {
        RefPtr<StorageAreaBase> storage;
        if (!m_path.isEmpty())
            storage = SQLiteStorageArea::create(localStorageQuotaInBytes, origin, m_path, WTFMove(workQueue));
        else
            storage = MemoryStorageArea::create(origin, StorageAreaBase::StorageType::Local);

        m_localStorageArea = storage;
        m_registry->registerStorageArea(storage->identifier(), *storage);
    }

    return *m_localStorageArea;
}

StorageAreaIdentifier LocalStorageManager::connectToLocalStorageArea(IPC::Connection::UniqueID connection, StorageAreaMapIdentifier sourceIdentifier, const WebCore::ClientOrigin& origin, Ref<WorkQueue>&& workQueue)
{
    Ref localStorageArea = ensureLocalStorageArea(origin, WTFMove(workQueue));
    ASSERT(m_path.isEmpty() || is<SQLiteStorageArea>(localStorageArea));
    localStorageArea->addListener(connection, sourceIdentifier);
    return localStorageArea->identifier();
}

MemoryStorageArea& LocalStorageManager::ensureTransientLocalStorageArea(const WebCore::ClientOrigin& origin)
{
    RefPtr transientStorageArea = m_transientStorageArea;
    if (!transientStorageArea) {
        transientStorageArea = MemoryStorageArea::create(origin, StorageAreaBase::StorageType::Local);
        m_transientStorageArea = transientStorageArea;
        m_registry->registerStorageArea(transientStorageArea->identifier(), *transientStorageArea);
    }

    return *m_transientStorageArea;
}

StorageAreaIdentifier LocalStorageManager::connectToTransientLocalStorageArea(IPC::Connection::UniqueID connection, StorageAreaMapIdentifier sourceIdentifier, const WebCore::ClientOrigin& origin)
{
    Ref transientStorageArea = ensureTransientLocalStorageArea(origin);
    ASSERT(is<MemoryStorageArea>(transientStorageArea));
    transientStorageArea->addListener(connection, sourceIdentifier);
    return transientStorageArea->identifier();
}

void LocalStorageManager::cancelConnectToLocalStorageArea(IPC::Connection::UniqueID connection)
{
    connectionClosedForLocalStorageArea(connection);
}

void LocalStorageManager::cancelConnectToTransientLocalStorageArea(IPC::Connection::UniqueID connection)
{
    connectionClosedForLocalStorageArea(connection);
}

void LocalStorageManager::disconnectFromStorageArea(IPC::Connection::UniqueID connection, StorageAreaIdentifier identifier)
{
    if (m_localStorageArea && m_localStorageArea->identifier() == identifier) {
        connectionClosedForLocalStorageArea(connection);
        return;
    }

    if (m_transientStorageArea && m_transientStorageArea->identifier() == identifier)
        connectionClosedForTransientStorageArea(connection);
}

HashMap<String, String> LocalStorageManager::fetchStorageMap() const
{
    if (RefPtr localStorageArea = m_localStorageArea)
        return localStorageArea->allItems();

    if (RefPtr transientStorageArea = m_transientStorageArea)
        return transientStorageArea->allItems();

    return { };
}

bool LocalStorageManager::setStorageMap(WebCore::ClientOrigin clientOrigin, HashMap<String, String>&& storageMap, Ref<WorkQueue>&& workQueue)
{
    bool succeeded = true;

    if (clientOrigin.topOrigin == clientOrigin.clientOrigin) {
        Ref localStorageArea = ensureLocalStorageArea(clientOrigin, WTFMove(workQueue));
        for (auto& [key, value] : storageMap) {
            if (!localStorageArea->setItem({ }, { }, WTFMove(key), WTFMove(value), { }))
                succeeded = false;
        }
    } else {
        Ref transientStorageArea = ensureTransientLocalStorageArea(clientOrigin);
        for (auto& [key, value] : storageMap) {
            if (!transientStorageArea->setItem({ }, { }, WTFMove(key), WTFMove(value), { }))
                succeeded = false;
        }
    }

    return succeeded;
}

} // namespace WebKit
