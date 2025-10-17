/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 31, 2022.
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

#include <WebCore/SecurityOriginData.h>
#include <WebCore/StorageArea.h>
#include <WebCore/StorageNamespace.h>
#include <pal/SessionID.h>
#include <wtf/CheckedRef.h>
#include <wtf/HashMap.h>
#include <wtf/RefPtr.h>
#include <wtf/text/WTFString.h>

namespace WebKit {

class StorageAreaImpl;

class StorageNamespaceImpl final : public WebCore::StorageNamespace, public CanMakeWeakPtr<StorageNamespaceImpl> {
public:
    static Ref<StorageNamespaceImpl> createSessionStorageNamespace(unsigned quota, PAL::SessionID);
    static Ref<StorageNamespaceImpl> getOrCreateLocalStorageNamespace(const String& databasePath, unsigned quota, PAL::SessionID);
    virtual ~StorageNamespaceImpl();

    void close();

    // Not removing the origin's StorageArea from m_storageAreaMap because
    // we're just deleting the underlying db file. If an item is added immediately
    // after file deletion, we want the same StorageArea to eventually trigger
    // a sync and for StorageAreaSync to recreate the backing db file.
    void clearOriginForDeletion(const WebCore::SecurityOriginData&);
    void clearAllOriginsForDeletion();
    void sync();
    void closeIdleLocalStorageDatabases();

    PAL::SessionID sessionID() const final { return m_sessionID; }
    void setSessionIDForTesting(PAL::SessionID) final;
    const WebCore::SecurityOrigin* topLevelOrigin() const final { return nullptr; };

private:
    StorageNamespaceImpl(WebCore::StorageType, const String& path, unsigned quota, PAL::SessionID);

    Ref<WebCore::StorageArea> storageArea(const WebCore::SecurityOrigin&) final;
    Ref<StorageNamespace> copy(WebCore::Page& newPage) final;

    typedef HashMap<WebCore::SecurityOriginData, RefPtr<StorageAreaImpl>> StorageAreaMap;
    StorageAreaMap m_storageAreaMap;

    WebCore::StorageType m_storageType;

    // Only used if m_storageType == LocalStorage and the path was not "" in our constructor.
    String m_path;
    RefPtr<WebCore::StorageSyncManager> m_syncManager;

    // The default quota for each new storage area.
    unsigned m_quota;

    bool m_isShutdown;

    PAL::SessionID m_sessionID;
};

} // namespace WebCore
