/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 8, 2022.
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
#include "SessionStorageManager.h"

#include "MemoryStorageArea.h"
#include "StorageAreaRegistry.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(SessionStorageManager);

SessionStorageManager::SessionStorageManager(StorageAreaRegistry& registry)
    : m_registry(registry)
{
}

bool SessionStorageManager::isActive() const
{
    return WTF::anyOf(m_storageAreas.values(), [&] (auto& storageArea) {
        return storageArea->hasListeners();
    });
}

bool SessionStorageManager::hasDataInMemory() const
{
    return WTF::anyOf(m_storageAreas.values(), [&] (auto& storageArea) {
        return !storageArea->isEmpty();
    });
}

void SessionStorageManager::clearData()
{
    for (Ref storageArea : m_storageAreas.values())
        storageArea->clear();
}

void SessionStorageManager::connectionClosed(IPC::Connection::UniqueID connection)
{
    for (Ref storageArea : m_storageAreas.values())
        storageArea->removeListener(connection);
}

void SessionStorageManager::removeNamespace(StorageNamespaceIdentifier namespaceIdentifier)
{
    auto identifier = m_storageAreasByNamespace.takeOptional(namespaceIdentifier);
    if (!identifier)
        return;

    m_storageAreas.remove(*identifier);
    m_registry->unregisterStorageArea(*identifier);
}

StorageAreaIdentifier SessionStorageManager::addStorageArea(Ref<MemoryStorageArea>&& storageArea, StorageNamespaceIdentifier namespaceIdentifier)
{
    auto identifier = storageArea->identifier();
    m_registry->registerStorageArea(identifier, storageArea);
    m_storageAreasByNamespace.add(namespaceIdentifier, identifier);
    m_storageAreas.add(identifier, WTFMove(storageArea));

    return identifier;
}

std::optional<StorageAreaIdentifier> SessionStorageManager::connectToSessionStorageArea(IPC::Connection::UniqueID connection, StorageAreaMapIdentifier sourceIdentifier, const WebCore::ClientOrigin& origin, StorageNamespaceIdentifier namespaceIdentifier)
{
    auto identifier = m_storageAreasByNamespace.getOptional(namespaceIdentifier);
    if (!identifier)
        identifier = addStorageArea(MemoryStorageArea::create(origin), namespaceIdentifier);

    RefPtr storageArea = m_storageAreas.get(*identifier);
    if (!storageArea)
        return std::nullopt;

    storageArea->addListener(connection, sourceIdentifier);

    return identifier;
}

void SessionStorageManager::cancelConnectToSessionStorageArea(IPC::Connection::UniqueID connection, StorageNamespaceIdentifier namespaceIdentifier)
{
    auto identifier = m_storageAreasByNamespace.getOptional(namespaceIdentifier);
    if (!identifier)
        return;

    RefPtr storageArea = m_storageAreas.get(*identifier);
    if (!storageArea)
        return;

    storageArea->removeListener(connection);
}

void SessionStorageManager::disconnectFromStorageArea(IPC::Connection::UniqueID connection, StorageAreaIdentifier identifier)
{
    if (RefPtr storageArea = m_storageAreas.get(identifier))
        storageArea->removeListener(connection);
}

void SessionStorageManager::cloneStorageArea(StorageNamespaceIdentifier sourceNamespaceIdentifier, StorageNamespaceIdentifier targetNamespaceIdentifier)
{
    auto identifier = m_storageAreasByNamespace.getOptional(sourceNamespaceIdentifier);
    if (!identifier)
        return;

    if (RefPtr storageArea = m_storageAreas.get(*identifier))
        addStorageArea(storageArea->clone(), targetNamespaceIdentifier);
}

HashMap<String, String> SessionStorageManager::fetchStorageMap(StorageNamespaceIdentifier namespaceIdentifier)
{
    auto identifier = m_storageAreasByNamespace.getOptional(namespaceIdentifier);
    if (!identifier)
        return { };

    RefPtr storageArea = m_storageAreas.get(*identifier);
    if (!storageArea)
        return { };

    return storageArea->allItems();
}

bool SessionStorageManager::setStorageMap(StorageNamespaceIdentifier storageNamespaceIdentifier, WebCore::ClientOrigin clientOrigin, HashMap<String, String>&& storageMap)
{
    auto identifier = m_storageAreasByNamespace.getOptional(storageNamespaceIdentifier);
    if (!identifier)
        identifier = addStorageArea(MemoryStorageArea::create(clientOrigin), storageNamespaceIdentifier);

    RefPtr storageArea = m_storageAreas.get(*identifier);
    if (!storageArea)
        return false;

    bool succeeded = true;
    for (auto& [key, value] : storageMap) {
        if (!storageArea->setItem({ }, { }, WTFMove(key), WTFMove(value), { }))
            succeeded = false;
    }

    return succeeded;
}

} // namespace WebKit
