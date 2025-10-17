/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 17, 2022.
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

#include "EventTarget.h"
#include "IDBActiveDOMObject.h"
#include "IDBConnectionProxy.h"
#include "IDBDatabaseConnectionIdentifier.h"
#include "IDBDatabaseInfo.h"
#include "IDBKeyPath.h"
#include "IDBTransactionMode.h"
#include <wtf/ThreadSafeWeakPtr.h>

namespace WebCore {

class DOMStringList;
class IDBObjectStore;
class IDBOpenDBRequest;
class IDBResultData;
class IDBTransaction;
class IDBTransactionInfo;

struct EventNames;

class IDBDatabase final : public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<IDBDatabase>, public EventTarget, public IDBActiveDOMObject {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(IDBDatabase);
public:
    static Ref<IDBDatabase> create(ScriptExecutionContext&, IDBClient::IDBConnectionProxy&, const IDBResultData&);

    virtual ~IDBDatabase();

    // IDBDatabase IDL
    const String name() const;
    uint64_t version() const;
    Ref<DOMStringList> objectStoreNames() const;

    struct ObjectStoreParameters {
        std::optional<IDBKeyPath> keyPath;
        bool autoIncrement;
    };

    ExceptionOr<Ref<IDBObjectStore>> createObjectStore(const String& name, ObjectStoreParameters&&);

    using StringOrVectorOfStrings = std::variant<String, Vector<String>>;
    struct TransactionOptions {
        std::optional<IDBTransactionDurability> durability;
    };
    ExceptionOr<Ref<IDBTransaction>> transaction(StringOrVectorOfStrings&& storeNames, IDBTransactionMode, TransactionOptions = { });
    ExceptionOr<void> deleteObjectStore(const String& name);
    void close();

    void renameObjectStore(IDBObjectStore&, const String& newName);
    void renameIndex(IDBIndex&, const String& newName);

    // EventTarget
    enum EventTargetInterfaceType eventTargetInterface() const final { return EventTargetInterfaceType::IDBDatabase; }
    ScriptExecutionContext* scriptExecutionContext() const final { return ActiveDOMObject::scriptExecutionContext(); }
    void refEventTarget() final { ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr::ref(); }
    void derefEventTarget() final { ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr::deref(); }

    // ActiveDOMObject.
    void ref() const final { ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr::ref(); }
    void deref() const final { ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr::deref(); }

    IDBDatabaseInfo& info() { return m_info; }
    IDBDatabaseConnectionIdentifier databaseConnectionIdentifier() const { return m_databaseConnectionIdentifier; }

    Ref<IDBTransaction> startVersionChangeTransaction(const IDBTransactionInfo&, IDBOpenDBRequest&);
    void didStartTransaction(IDBTransaction&);

    void willCommitTransaction(IDBTransaction&);
    void didCommitTransaction(IDBTransaction&);
    void willAbortTransaction(IDBTransaction&);
    void didAbortTransaction(IDBTransaction&);

    void fireVersionChangeEvent(const IDBResourceIdentifier& requestIdentifier, uint64_t requestedVersion);
    void didCloseFromServer(const IDBError&);
    void connectionToServerLost(const IDBError&);

    IDBClient::IDBConnectionProxy& connectionProxy() { return m_connectionProxy.get(); }

    void didCreateIndexInfo(const IDBIndexInfo&);
    void didDeleteIndexInfo(const IDBIndexInfo&);

    bool isClosingOrClosed() const { return m_closePending || m_closedInServer; }

    void dispatchEvent(Event&) final;

    void setIsContextSuspended(bool isContextSuspended) { m_isContextSuspended = isContextSuspended; }
    bool isContextSuspended() const { return m_isContextSuspended; }

private:
    IDBDatabase(ScriptExecutionContext&, IDBClient::IDBConnectionProxy&, const IDBResultData&);

    void didCommitOrAbortTransaction(IDBTransaction&);

    // ActiveDOMObject.
    bool virtualHasPendingActivity() const final;
    void stop() final;

    void maybeCloseInServer();

    Ref<IDBClient::IDBConnectionProxy> m_connectionProxy;
    IDBDatabaseInfo m_info;
    IDBDatabaseConnectionIdentifier m_databaseConnectionIdentifier;

    bool m_closePending { false };
    bool m_closedInServer { false };

    RefPtr<IDBTransaction> m_versionChangeTransaction;
    HashMap<IDBResourceIdentifier, RefPtr<IDBTransaction>> m_activeTransactions;
    HashMap<IDBResourceIdentifier, RefPtr<IDBTransaction>> m_committingTransactions;
    HashMap<IDBResourceIdentifier, RefPtr<IDBTransaction>> m_abortingTransactions;
    
    const EventNames& m_eventNames; // Need to cache this so we can use it from GC threads.
    std::atomic<bool> m_isContextSuspended { false };
};

} // namespace WebCore
