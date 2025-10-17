/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 22, 2023.
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
#include "StorageAreaImplIdentifier.h"
#include "StorageAreaMapIdentifier.h"
#include <WebCore/ClientOrigin.h>
#include <wtf/HashMap.h>
#include <wtf/Identified.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
struct ClientOrigin;
}

namespace WebKit {

enum class StorageError : uint8_t {
    Database,
    ItemNotFound,
    QuotaExceeded,
};

class StorageAreaBase : public CanMakeWeakPtr<StorageAreaBase>, public Identified<StorageAreaIdentifier> {
    WTF_MAKE_TZONE_ALLOCATED(StorageAreaBase);
    WTF_MAKE_NONCOPYABLE(StorageAreaBase);
public:
    static uint64_t nextMessageIdentifier();
    virtual ~StorageAreaBase();

    enum class Type : bool { SQLite, Memory };
    virtual Type type() const = 0;
    enum class StorageType : bool { Session, Local };
    virtual StorageType storageType() const = 0;
    virtual bool isEmpty() = 0;
    virtual void clear() = 0;

    WebCore::ClientOrigin origin() const { return m_origin; }
    unsigned quota() const { return m_quota; }
    void addListener(IPC::Connection::UniqueID, StorageAreaMapIdentifier);
    void removeListener(IPC::Connection::UniqueID);
    bool hasListeners() const { return !m_listeners.isEmpty(); }
    void notifyListenersAboutClear();

    virtual HashMap<String, String> allItems() = 0;
    virtual Expected<void, StorageError> setItem(std::optional<IPC::Connection::UniqueID>, std::optional<StorageAreaImplIdentifier>, String&& key, String&& value, const String& urlString) = 0;
    virtual Expected<void, StorageError> removeItem(IPC::Connection::UniqueID, StorageAreaImplIdentifier, const String& key, const String& urlString) = 0;
    virtual Expected<void, StorageError> clear(IPC::Connection::UniqueID, StorageAreaImplIdentifier, const String& urlString) = 0;

    virtual void ref() const = 0;
    virtual void deref() const = 0;

protected:
    StorageAreaBase(unsigned quota, const WebCore::ClientOrigin&);
    void dispatchEvents(IPC::Connection::UniqueID, StorageAreaImplIdentifier, const String& key, const String& oldValue, const String& newValue, const String& urlString) const;

private:
    unsigned m_quota;
    WebCore::ClientOrigin m_origin;
    HashMap<IPC::Connection::UniqueID, StorageAreaMapIdentifier> m_listeners;
};

} // namespace WebKit
