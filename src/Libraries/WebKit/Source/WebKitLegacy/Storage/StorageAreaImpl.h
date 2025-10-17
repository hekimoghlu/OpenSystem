/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 14, 2025.
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
#include <WebCore/StorageMap.h>
#include <WebCore/Timer.h>
#include <wtf/HashMap.h>
#include <wtf/RefPtr.h>

namespace WebCore {
class SecurityOrigin;
class StorageMap;
}

namespace WebKit {

class StorageAreaSync;

class StorageAreaImpl : public WebCore::StorageArea {
public:
    static Ref<StorageAreaImpl> create(WebCore::StorageType, const WebCore::SecurityOrigin&, RefPtr<WebCore::StorageSyncManager>&&, unsigned quota);
    virtual ~StorageAreaImpl();

    unsigned length() override;
    String key(unsigned index) override;
    String item(const String& key) override;
    void setItem(WebCore::LocalFrame& sourceFrame, const String& key, const String& value, bool& quotaException) override;
    void removeItem(WebCore::LocalFrame& sourceFrame, const String& key) override;
    void clear(WebCore::LocalFrame& sourceFrame) override;
    bool contains(const String& key) override;

    WebCore::StorageType storageType() const override;

    size_t memoryBytesUsedByCache() override;

    void incrementAccessCount() override;
    void decrementAccessCount() override;
    void closeDatabaseIfIdle() override;

    Ref<StorageAreaImpl> copy();
    void close();

    // Only called from a background thread.
    void importItems(HashMap<String, String>&& items);

    // Used to clear a StorageArea and close db before backing db file is deleted.
    void clearForOriginDeletion();

    void sync();

    void sessionChanged(bool isNewSessionPersistent);

private:
    StorageAreaImpl(WebCore::StorageType, const WebCore::SecurityOrigin&, RefPtr<WebCore::StorageSyncManager>&&, unsigned quota);
    explicit StorageAreaImpl(const StorageAreaImpl&);

    void blockUntilImportComplete() const;
    void closeDatabaseTimerFired();

    void dispatchStorageEvent(const String& key, const String& oldValue, const String& newValue, WebCore::LocalFrame& sourceFrame);

    WebCore::StorageType m_storageType;
    Ref<const WebCore::SecurityOrigin> m_securityOrigin;
    WebCore::StorageMap m_storageMap;

    RefPtr<StorageAreaSync> m_storageAreaSync;
    RefPtr<WebCore::StorageSyncManager> m_storageSyncManager;

#if ASSERT_ENABLED
    bool m_isShutdown { false };
#endif
    unsigned m_accessCount;
    WebCore::Timer m_closeDatabaseTimer;
};

} // namespace WebCore
