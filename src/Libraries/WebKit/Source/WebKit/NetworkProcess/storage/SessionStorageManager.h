/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 27, 2025.
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

#include "Connection.h"
#include "StorageAreaIdentifier.h"
#include "StorageAreaMapIdentifier.h"
#include "StorageNamespaceIdentifier.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {
struct ClientOrigin;
} // namespace WebCore

namespace WebKit {

class MemoryStorageArea;
class StorageAreaRegistry;

class SessionStorageManager {
    WTF_MAKE_TZONE_ALLOCATED(SessionStorageManager);
public:
    explicit SessionStorageManager(StorageAreaRegistry&);
    bool isActive() const;
    bool hasDataInMemory() const;
    void clearData();
    void connectionClosed(IPC::Connection::UniqueID);
    void removeNamespace(StorageNamespaceIdentifier);

    std::optional<StorageAreaIdentifier> connectToSessionStorageArea(IPC::Connection::UniqueID, StorageAreaMapIdentifier, const WebCore::ClientOrigin&, StorageNamespaceIdentifier);
    void cancelConnectToSessionStorageArea(IPC::Connection::UniqueID, StorageNamespaceIdentifier);
    void disconnectFromStorageArea(IPC::Connection::UniqueID, StorageAreaIdentifier);
    void cloneStorageArea(StorageNamespaceIdentifier, StorageNamespaceIdentifier);

    HashMap<String, String> fetchStorageMap(StorageNamespaceIdentifier);
    bool setStorageMap(StorageNamespaceIdentifier, WebCore::ClientOrigin, HashMap<String, String>&&);

private:
    StorageAreaIdentifier addStorageArea(Ref<MemoryStorageArea>&&, StorageNamespaceIdentifier);

    CheckedRef<StorageAreaRegistry> m_registry;
    HashMap<StorageAreaIdentifier, Ref<MemoryStorageArea>> m_storageAreas;
    HashMap<StorageNamespaceIdentifier, StorageAreaIdentifier> m_storageAreasByNamespace;
};

} // namespace WebKit
