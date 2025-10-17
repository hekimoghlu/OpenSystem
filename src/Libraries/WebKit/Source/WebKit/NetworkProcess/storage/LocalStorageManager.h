/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 14, 2021.
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
#include <wtf/TZoneMalloc.h>
#include <wtf/WorkQueue.h>

namespace WebCore {
struct ClientOrigin;
class SecurityOriginData;
}

namespace WebKit {

class MemoryStorageArea;
class StorageAreaBase;
class StorageAreaRegistry;

class LocalStorageManager {
    WTF_MAKE_TZONE_ALLOCATED(LocalStorageManager);
public:
    static Vector<WebCore::SecurityOriginData> originsOfLocalStorageData(const String& path);
    static String localStorageFilePath(const String& directory, const WebCore::ClientOrigin&);
    static String localStorageFilePath(const String& directory);

    LocalStorageManager(const String& path, StorageAreaRegistry&);
    bool isActive() const;
    bool hasDataInMemory() const;
    void clearDataInMemory();
    void clearDataOnDisk();
    void close();
    void handleLowMemoryWarning();
    void syncLocalStorage();
    void connectionClosed(IPC::Connection::UniqueID);

    StorageAreaIdentifier connectToLocalStorageArea(IPC::Connection::UniqueID, StorageAreaMapIdentifier, const WebCore::ClientOrigin&, Ref<WorkQueue>&&);
    StorageAreaIdentifier connectToTransientLocalStorageArea(IPC::Connection::UniqueID, StorageAreaMapIdentifier, const WebCore::ClientOrigin&);
    void cancelConnectToLocalStorageArea(IPC::Connection::UniqueID);
    void cancelConnectToTransientLocalStorageArea(IPC::Connection::UniqueID);
    void disconnectFromStorageArea(IPC::Connection::UniqueID, StorageAreaIdentifier);

    HashMap<String, String> fetchStorageMap() const;
    bool setStorageMap(WebCore::ClientOrigin, HashMap<String, String>&&, Ref<WorkQueue>&&);

private:
    void connectionClosedForLocalStorageArea(IPC::Connection::UniqueID);
    void connectionClosedForTransientStorageArea(IPC::Connection::UniqueID);

    StorageAreaBase& ensureLocalStorageArea(const WebCore::ClientOrigin&, Ref<WorkQueue>&&);
    MemoryStorageArea& ensureTransientLocalStorageArea(const WebCore::ClientOrigin&);

    String m_path;
    CheckedRef<StorageAreaRegistry> m_registry;
    RefPtr<MemoryStorageArea> m_transientStorageArea;
    RefPtr<StorageAreaBase> m_localStorageArea;
};

} // namespace WebKit


