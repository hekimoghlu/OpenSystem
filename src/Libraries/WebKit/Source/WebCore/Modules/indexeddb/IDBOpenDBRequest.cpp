/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 10, 2021.
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
#include "IDBOpenDBRequest.h"

#include "DOMException.h"
#include "EventNames.h"
#include "IDBConnectionProxy.h"
#include "IDBConnectionToServer.h"
#include "IDBDatabase.h"
#include "IDBError.h"
#include "IDBOpenRequestData.h"
#include "IDBRequestCompletionEvent.h"
#include "IDBResultData.h"
#include "IDBTransaction.h"
#include "IDBVersionChangeEvent.h"
#include "Logging.h"
#include "ScriptExecutionContext.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(IDBOpenDBRequest);

Ref<IDBOpenDBRequest> IDBOpenDBRequest::createDeleteRequest(ScriptExecutionContext& context, IDBClient::IDBConnectionProxy& connectionProxy, const IDBDatabaseIdentifier& databaseIdentifier)
{
    auto result = adoptRef(*new IDBOpenDBRequest(context, connectionProxy, databaseIdentifier, 0, IndexedDB::RequestType::Delete));
    result->suspendIfNeeded();
    return result;
}

Ref<IDBOpenDBRequest> IDBOpenDBRequest::createOpenRequest(ScriptExecutionContext& context, IDBClient::IDBConnectionProxy& connectionProxy, const IDBDatabaseIdentifier& databaseIdentifier, uint64_t version)
{
    auto result = adoptRef(*new IDBOpenDBRequest(context, connectionProxy, databaseIdentifier, version, IndexedDB::RequestType::Open));
    result->suspendIfNeeded();
    return result;
}
    
IDBOpenDBRequest::IDBOpenDBRequest(ScriptExecutionContext& context, IDBClient::IDBConnectionProxy& connectionProxy, const IDBDatabaseIdentifier& databaseIdentifier, uint64_t version, IndexedDB::RequestType requestType)
    : IDBRequest(context, connectionProxy, requestType)
    , m_databaseIdentifier(databaseIdentifier)
    , m_version(version)
{
}

IDBOpenDBRequest::~IDBOpenDBRequest()
{
    ASSERT(canCurrentThreadAccessThreadLocalData(originThread()));
}

void IDBOpenDBRequest::onError(const IDBResultData& data)
{
    ASSERT(canCurrentThreadAccessThreadLocalData(originThread()));

    m_domError = data.error().toDOMException();
    enqueueEvent(IDBRequestCompletionEvent::create(eventNames().errorEvent, Event::CanBubble::Yes, Event::IsCancelable::Yes, *this));
}

void IDBOpenDBRequest::versionChangeTransactionDidFinish()
{
    ASSERT(canCurrentThreadAccessThreadLocalData(originThread()));

    // 3.3.7 "versionchange" transaction steps
    // When the transaction is finished, after firing complete/abort on the transaction, immediately set request's transaction property to null.
    setShouldExposeTransactionToDOM(false);
}

void IDBOpenDBRequest::fireSuccessAfterVersionChangeCommit()
{
    LOG(IndexedDB, "IDBOpenDBRequest::fireSuccessAfterVersionChangeCommit() - %s", resourceIdentifier().loggingString().utf8().data());

    ASSERT(canCurrentThreadAccessThreadLocalData(originThread()));
    ASSERT(hasPendingActivity());
    m_transaction->addRequest(*this);

    Ref event = IDBRequestCompletionEvent::create(eventNames().successEvent, Event::CanBubble::No, Event::IsCancelable::No, *this);
    m_openDatabaseSuccessEvent = event.get();

    enqueueEvent(WTFMove(event));
}

void IDBOpenDBRequest::fireErrorAfterVersionChangeCompletion()
{
    LOG(IndexedDB, "IDBOpenDBRequest::fireErrorAfterVersionChangeCompletion() - %s", resourceIdentifier().loggingString().utf8().data());

    ASSERT(canCurrentThreadAccessThreadLocalData(originThread()));
    ASSERT(hasPendingActivity());

    IDBError idbError(ExceptionCode::AbortError);
    m_domError = DOMException::create(ExceptionCode::AbortError);
    setResultToUndefined();

    m_transaction->addRequest(*this);
    enqueueEvent(IDBRequestCompletionEvent::create(eventNames().errorEvent, Event::CanBubble::Yes, Event::IsCancelable::Yes, *this));
}

void IDBOpenDBRequest::cancelForStop()
{
    connectionProxy().openDBRequestCancelled({ connectionProxy(), *this });
}

void IDBOpenDBRequest::dispatchEvent(Event& event)
{
    ASSERT(canCurrentThreadAccessThreadLocalData(originThread()));

    Ref protectedThis { *this };

    IDBRequest::dispatchEvent(event);

    if (m_transaction && m_transaction->isVersionChange() && (event.type() == eventNames().errorEvent || event.type() == eventNames().successEvent))
        m_transaction->database().connectionProxy().didFinishHandlingVersionChangeTransaction(m_transaction->database().databaseConnectionIdentifier(), *m_transaction);
}

void IDBOpenDBRequest::onSuccess(const IDBResultData& resultData)
{
    LOG(IndexedDB, "IDBOpenDBRequest::onSuccess()");

    ASSERT(canCurrentThreadAccessThreadLocalData(originThread()));

    setResult(IDBDatabase::create(*scriptExecutionContext(), connectionProxy(), resultData));
    setReadyState(ReadyState::Done);

    enqueueEvent(IDBRequestCompletionEvent::create(eventNames().successEvent, Event::CanBubble::No, Event::IsCancelable::No, *this));
}

void IDBOpenDBRequest::onUpgradeNeeded(const IDBResultData& resultData)
{
    ASSERT(canCurrentThreadAccessThreadLocalData(originThread()));

    Ref<IDBDatabase> database = IDBDatabase::create(*scriptExecutionContext(), connectionProxy(), resultData);
    Ref<IDBTransaction> transaction = database->startVersionChangeTransaction(resultData.transactionInfo(), *this);

    ASSERT(transaction->info().mode() == IDBTransactionMode::Versionchange);
    ASSERT(transaction->originalDatabaseInfo());

    uint64_t oldVersion = transaction->originalDatabaseInfo()->version();
    uint64_t newVersion = transaction->info().newVersion();

    LOG(IndexedDB, "IDBOpenDBRequest::onUpgradeNeeded() - current version is %" PRIu64 ", new is %" PRIu64, oldVersion, newVersion);

    setResult(WTFMove(database));
    setReadyState(ReadyState::Done);
    m_transaction = WTFMove(transaction);
    m_transaction->addRequest(*this);

    enqueueEvent(IDBVersionChangeEvent::create(oldVersion, newVersion, eventNames().upgradeneededEvent));
}

void IDBOpenDBRequest::onDeleteDatabaseSuccess(const IDBResultData& resultData)
{
    ASSERT(canCurrentThreadAccessThreadLocalData(originThread()));

    uint64_t oldVersion = resultData.databaseInfo().version();

    LOG(IndexedDB, "IDBOpenDBRequest::onDeleteDatabaseSuccess() - current version is %" PRIu64, oldVersion);

    setReadyState(ReadyState::Done);
    setResultToUndefined();

    enqueueEvent(IDBVersionChangeEvent::create(oldVersion, 0, eventNames().successEvent));
}

void IDBOpenDBRequest::requestCompleted(const IDBResultData& data)
{
    LOG(IndexedDB, "IDBOpenDBRequest::requestCompleted");

    ASSERT(canCurrentThreadAccessThreadLocalData(originThread()));

    m_isBlocked = false;

    // If an Open request was completed after the page has navigated, leaving this request
    // with a stopped script execution context, we need to message back to the server so it
    // doesn't hang waiting on a database connection or transaction that will never exist.
    if (isContextStopped()) {
        switch (data.type()) {
        case IDBResultType::OpenDatabaseSuccess:
            connectionProxy().abortOpenAndUpgradeNeeded(data.databaseConnectionIdentifier(), std::nullopt);
            break;
        case IDBResultType::OpenDatabaseUpgradeNeeded:
            connectionProxy().abortOpenAndUpgradeNeeded(data.databaseConnectionIdentifier(), data.transactionInfo().identifier());
            break;
        default:
            break;
        }

        return;
    }

    switch (data.type()) {
    case IDBResultType::Error:
        onError(data);
        break;
    case IDBResultType::OpenDatabaseSuccess:
        onSuccess(data);
        break;
    case IDBResultType::OpenDatabaseUpgradeNeeded:
        onUpgradeNeeded(data);
        break;
    case IDBResultType::DeleteDatabaseSuccess:
        onDeleteDatabaseSuccess(data);
        break;
    default:
        RELEASE_ASSERT_NOT_REACHED();
    }
}

void IDBOpenDBRequest::requestBlocked(uint64_t oldVersion, uint64_t newVersion)
{
    ASSERT(canCurrentThreadAccessThreadLocalData(originThread()));
    ASSERT(!m_isBlocked);

    m_isBlocked = true;

    LOG(IndexedDB, "IDBOpenDBRequest::requestBlocked");
    enqueueEvent(IDBVersionChangeEvent::create(oldVersion, newVersion, eventNames().blockedEvent));
}

void IDBOpenDBRequest::setIsContextSuspended(bool isContextSuspended)
{
    m_isContextSuspended = isContextSuspended;

    // If this request is blocked, it means this request is being processed on the server.
    // The client needs to actively stop the request so it doesn't blocks the processing of subsequent requests.
    if (m_isBlocked) {
        IDBOpenRequestData requestData(connectionProxy(), *this);
        connectionProxy().openDBRequestCancelled(requestData);
        auto result = IDBResultData::error(requestData.requestIdentifier(), IDBError { ExceptionCode::UnknownError, "Blocked open request on cached page is aborted to unblock other requests"_s });
        requestCompleted(result);
    }
}

} // namespace WebCore
