/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 21, 2022.
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

#include "IDBIndexIdentifier.h"
#include "IDBObjectStoreIdentifier.h"
#include "IDBRequest.h"
#include "IDBRequestData.h"
#include "IDBResourceIdentifier.h"
#include "IDBResultData.h"
#include "IDBTransaction.h"
#include <optional>
#include <wtf/Function.h>
#include <wtf/MainThread.h>
#include <wtf/Markable.h>
#include <wtf/Threading.h>

namespace WebCore {

class IDBResultData;

namespace IndexedDB {
enum class IndexRecordType : bool;
}

namespace IDBClient {

class TransactionOperation : public ThreadSafeRefCounted<TransactionOperation> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(TransactionOperation);
    friend IDBRequestData::IDBRequestData(TransactionOperation&);
public:
    virtual ~TransactionOperation()
    {
        ASSERT(canCurrentThreadAccessThreadLocalData(originThread()));
    }

    void perform()
    {
        ASSERT(canCurrentThreadAccessThreadLocalData(originThread()));
        ASSERT(m_performFunction);
        m_performFunction();
        m_performFunction = { };
    }

    void transitionToCompleteOnThisThread(const IDBResultData& data)
    {
        ASSERT(canCurrentThreadAccessThreadLocalData(originThread()));
        m_transaction->operationCompletedOnServer(data, *this);
    }

    void transitionToComplete(const IDBResultData& data, RefPtr<TransactionOperation>&& lastRef)
    {
        ASSERT(isMainThread());

        if (canCurrentThreadAccessThreadLocalData(originThread()))
            transitionToCompleteOnThisThread(data);
        else {
            m_transaction->performCallbackOnOriginThread(*this, &TransactionOperation::transitionToCompleteOnThisThread, data);
            m_transaction->callFunctionOnOriginThread([lastRef = WTFMove(lastRef)]() {
            });
        }
    }

    void doComplete(const IDBResultData& data)
    {
        ASSERT(canCurrentThreadAccessThreadLocalData(originThread()));

        if (m_performFunction)
            m_performFunction = { };

        // Due to race conditions between the server sending an "operation complete" message and the client
        // forcefully aborting an operation, it's unavoidable that this method might be called twice.
        // It's okay to handle that gracefully with an early return.
        if (m_didComplete)
            return;
        m_didComplete = true;

        if (m_completeFunction) {
            m_completeFunction(data);
            // m_completeFunction should not hold ref to this TransactionOperation after its execution.
            m_completeFunction = { };
        }
        m_transaction->operationCompletedOnClient(*this);
    }

    const IDBResourceIdentifier& identifier() const { return m_identifier; }

    Thread& originThread() const { return m_originThread.get(); }

    IDBRequest* idbRequest() { return m_idbRequest.get(); }

    bool nextRequestCanGoToServer() const { return m_nextRequestCanGoToServer && m_idbRequest; }
    void setNextRequestCanGoToServer(bool nextRequestCanGoToServer) { m_nextRequestCanGoToServer = nextRequestCanGoToServer; }

    uint64_t operationID() const { return m_operationID; }

protected:
    TransactionOperation(IDBTransaction& transaction)
        : m_transaction(transaction)
        , m_identifier(transaction.connectionProxy())
        , m_operationID(transaction.generateOperationID())
    {
    }

    TransactionOperation(IDBTransaction&, IDBRequest&);

    Ref<IDBTransaction> m_transaction;
    IDBResourceIdentifier m_identifier;
    Markable<IDBObjectStoreIdentifier> m_objectStoreIdentifier;
    Markable<IDBIndexIdentifier> m_indexIdentifier;
    std::optional<IDBResourceIdentifier> m_cursorIdentifier;
    IndexedDB::IndexRecordType m_indexRecordType { IndexedDB::IndexRecordType::Key };
    Function<void()> m_performFunction;
    Function<void(const IDBResultData&)> m_completeFunction;

private:
    IDBResourceIdentifier transactionIdentifier() const { return m_transaction->info().identifier(); }
    std::optional<IDBObjectStoreIdentifier> objectStoreIdentifier() const { return m_objectStoreIdentifier; }
    std::optional<IDBIndexIdentifier> indexIdentifier() const { return m_indexIdentifier; }
    std::optional<IDBResourceIdentifier> cursorIdentifier() const { return m_cursorIdentifier; }
    IDBTransaction& transaction() { return m_transaction.get(); }
    IndexedDB::IndexRecordType indexRecordType() const { return m_indexRecordType; }

    Ref<Thread> m_originThread { Thread::current() };
    RefPtr<IDBRequest> m_idbRequest;
    bool m_nextRequestCanGoToServer { true };
    bool m_didComplete { false };

    uint64_t m_operationID { 0 };
};

class TransactionOperationImpl final : public TransactionOperation {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(TransactionOperationImpl);
public:
    template<typename... Args> static Ref<TransactionOperationImpl> create(Args&&... args) { return adoptRef(*new TransactionOperationImpl(std::forward<Args>(args)...)); }
private:
    TransactionOperationImpl(IDBTransaction& transaction, Function<void(const IDBResultData&)> completeMethod, Function<void(TransactionOperation&)> performMethod)
        : TransactionOperation(transaction)
    {
        ASSERT(performMethod);
        m_performFunction = [protectedThis = Ref { *this }, performMethod = WTFMove(performMethod)] {
            performMethod(protectedThis.get());
        };

        if (completeMethod) {
            m_completeFunction = [protectedThis = Ref { *this }, completeMethod = WTFMove(completeMethod)] (const IDBResultData& resultData) {
                completeMethod(resultData);
            };
        }
    }

    TransactionOperationImpl(IDBTransaction& transaction, IDBRequest& request, Function<void(const IDBResultData&)> completeMethod, Function<void(TransactionOperation&)> performMethod)
        : TransactionOperation(transaction, request)
    {
        ASSERT(performMethod);
        m_performFunction = [protectedThis = Ref { *this }, performMethod = WTFMove(performMethod)] {
            performMethod(protectedThis.get());
        };

        if (completeMethod) {
            m_completeFunction = [protectedThis = Ref { *this }, completeMethod = WTFMove(completeMethod)] (const IDBResultData& resultData) {
                completeMethod(resultData);
            };
        }
    }
};

} // namespace IDBClient
} // namespace WebCore
