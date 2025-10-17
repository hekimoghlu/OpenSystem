/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 4, 2022.
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

#include "MessageReceiver.h"
#include "StorageAreaIdentifier.h"
#include "StorageAreaImplIdentifier.h"
#include "StorageAreaMapIdentifier.h"
#include <WebCore/SecurityOrigin.h>
#include <WebCore/StorageArea.h>
#include <wtf/Forward.h>
#include <wtf/HashCountedSet.h>
#include <wtf/Identified.h>
#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
class SecurityOrigin;
class StorageMap;
struct ClientOrigin;
}

namespace WebKit {

class StorageAreaImpl;
class StorageNamespaceImpl;

class StorageAreaMap final : public RefCounted<StorageAreaMap>, public IPC::MessageReceiver, public Identified<StorageAreaMapIdentifier> {
    WTF_MAKE_TZONE_ALLOCATED(StorageAreaMap);
public:
    static Ref<StorageAreaMap> create(StorageNamespaceImpl& storageNamespace, Ref<const WebCore::SecurityOrigin>&& securityOrigin)
    {
        return adoptRef(*new StorageAreaMap(storageNamespace, WTFMove(securityOrigin)));
    }

    ~StorageAreaMap();

    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    WebCore::StorageType type() const { return m_type; }

    unsigned length();
    String key(unsigned index);
    String item(const String& key);
    void setItem(WebCore::LocalFrame& sourceFrame, StorageAreaImpl* sourceArea, const String& key, const String& value, bool& quotaException);
    void removeItem(WebCore::LocalFrame& sourceFrame, StorageAreaImpl* sourceArea, const String& key);
    void clear(WebCore::LocalFrame& sourceFrame, StorageAreaImpl* sourceArea);
    bool contains(const String& key);

    // IPC::MessageReceiver
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;

    const WebCore::SecurityOrigin& securityOrigin() const { return m_securityOrigin.get(); }

    void connect();
    void disconnect();
    void incrementUseCount();
    void decrementUseCount();

private:
    StorageAreaMap(StorageNamespaceImpl&, Ref<const WebCore::SecurityOrigin>&&);

    void didSetItem(uint64_t mapSeed, const String& key, bool hasError, HashMap<String, String>&&);
    void didRemoveItem(uint64_t mapSeed, const String& key, bool hasError, HashMap<String, String>&&);
    void didClear(uint64_t mapSeed);

    // Message handlers.
    void dispatchStorageEvent(const std::optional<StorageAreaImplIdentifier>& sourceStorageAreaID, const String& key, const String& oldValue, const String& newValue, const String& urlString, uint64_t messageIdentifier);
    void clearCache(uint64_t messageIdentifier);

    void syncOneItem(const String& key, const String& value);
    void syncItems(HashMap<String, String>&&);
    WebCore::StorageMap& ensureMap();
    WebCore::StorageType computeStorageType() const;
    WebCore::ClientOrigin clientOrigin() const;

    void applyChange(const String& key, const String& newValue);
    void dispatchSessionStorageEvent(const std::optional<StorageAreaImplIdentifier>&, const String& key, const String& oldValue, const String& newValue, const String& urlString);
    void dispatchLocalStorageEvent(const std::optional<StorageAreaImplIdentifier>&, const String& key, const String& oldValue, const String& newValue, const String& urlString);

    enum class SendMode : bool { Async, Sync };
    void sendConnectMessage(SendMode);
    void connectSync();
    void didConnect(std::optional<StorageAreaIdentifier>, HashMap<String, String>&&, uint64_t messageIdentifier);

    Ref<StorageNamespaceImpl> protectedNamespace() const;

    uint64_t m_lastHandledMessageIdentifier { 0 };
    WeakRef<StorageNamespaceImpl> m_namespace;
    Ref<const WebCore::SecurityOrigin> m_securityOrigin;
    std::unique_ptr<WebCore::StorageMap> m_map;
    std::optional<StorageAreaIdentifier> m_remoteAreaIdentifier;
    HashCountedSet<String> m_pendingValueChanges;
    uint64_t m_currentSeed { 1 };
    unsigned m_quotaInBytes;
    WebCore::StorageType m_type;
    uint64_t m_useCount { 0 };
    bool m_hasPendingClear { false };
    bool m_isWaitingForConnectReply { false };
};

} // namespace WebKit
