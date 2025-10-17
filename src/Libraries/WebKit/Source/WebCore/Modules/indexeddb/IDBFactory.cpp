/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 27, 2025.
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
#include "IDBFactory.h"

#include "Document.h"
#include "FrameDestructionObserverInlines.h"
#include "IDBBindingUtilities.h"
#include "IDBConnectionProxy.h"
#include "IDBDatabaseIdentifier.h"
#include "IDBKey.h"
#include "IDBOpenDBRequest.h"
#include "JSDOMPromiseDeferred.h"
#include "JSIDBFactory.h"
#include "Logging.h"
#include "Page.h"
#include "ScriptExecutionContext.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {
using namespace JSC;

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(IDBFactory);

static bool shouldThrowSecurityException(ScriptExecutionContext& context)
{
    ASSERT(is<Document>(context) || context.isWorkerGlobalScope());
    if (auto* document = dynamicDowncast<Document>(context)) {
        if (!document->frame())
            return true;
        if (!document->page())
            return true;
    }

    return context.canAccessResource(ScriptExecutionContext::ResourceType::IndexedDB) == ScriptExecutionContext::HasResourceAccess::No;
}

Ref<IDBFactory> IDBFactory::create(IDBClient::IDBConnectionProxy& connectionProxy)
{
    return adoptRef(*new IDBFactory(connectionProxy));
}

IDBFactory::IDBFactory(IDBClient::IDBConnectionProxy& connectionProxy)
    : m_connectionProxy(connectionProxy)
{
}

IDBFactory::~IDBFactory() = default;

ExceptionOr<Ref<IDBOpenDBRequest>> IDBFactory::open(ScriptExecutionContext& context, const String& name, std::optional<uint64_t> version)
{
    LOG(IndexedDB, "IDBFactory::open");
    
    if (version && !version.value())
        return Exception { ExceptionCode::TypeError, "IDBFactory.open() called with a version of 0"_s };

    return openInternal(context, name, version.value_or(0));
}

ExceptionOr<Ref<IDBOpenDBRequest>> IDBFactory::openInternal(ScriptExecutionContext& context, const String& name, uint64_t version)
{
    if (name.isNull())
        return Exception { ExceptionCode::TypeError, "IDBFactory.open() called without a database name"_s };

    if (shouldThrowSecurityException(context))
        return Exception { ExceptionCode::SecurityError, "IDBFactory.open() called in an invalid security context"_s };

    ASSERT(context.securityOrigin());
    bool isTransient = (context.canAccessResource(ScriptExecutionContext::ResourceType::IndexedDB) == ScriptExecutionContext::HasResourceAccess::DefaultForThirdParty);
    IDBDatabaseIdentifier databaseIdentifier(name, SecurityOriginData { context.securityOrigin()->data() }, SecurityOriginData { context.topOrigin().data() }, isTransient);
    if (!databaseIdentifier.isValid())
        return Exception { ExceptionCode::TypeError, "IDBFactory.open() called with an invalid security origin"_s };

    LOG(IndexedDBOperations, "IDB opening database: %s %" PRIu64, name.utf8().data(), version);

    return m_connectionProxy->openDatabase(context, databaseIdentifier, version);
}

ExceptionOr<Ref<IDBOpenDBRequest>> IDBFactory::deleteDatabase(ScriptExecutionContext& context, const String& name)
{
    LOG(IndexedDB, "IDBFactory::deleteDatabase - %s", name.utf8().data());

    if (name.isNull())
        return Exception { ExceptionCode::TypeError, "IDBFactory.deleteDatabase() called without a database name"_s };

    if (shouldThrowSecurityException(context))
        return Exception { ExceptionCode::SecurityError, "IDBFactory.deleteDatabase() called in an invalid security context"_s };

    ASSERT(context.securityOrigin());
    bool isTransient = (context.canAccessResource(ScriptExecutionContext::ResourceType::IndexedDB) == ScriptExecutionContext::HasResourceAccess::DefaultForThirdParty);
    IDBDatabaseIdentifier databaseIdentifier(name, SecurityOriginData { context.securityOrigin()->data() }, SecurityOriginData { context.topOrigin().data() }, isTransient);
    if (!databaseIdentifier.isValid())
        return Exception { ExceptionCode::TypeError, "IDBFactory.deleteDatabase() called with an invalid security origin"_s };

    LOG(IndexedDBOperations, "IDB deleting database: %s", name.utf8().data());

    return m_connectionProxy->deleteDatabase(context, databaseIdentifier);
}

ExceptionOr<short> IDBFactory::cmp(JSGlobalObject& execState, JSValue firstValue, JSValue secondValue)
{
    auto first = scriptValueToIDBKey(execState, firstValue);
    if (!first->isValid())
        return Exception { ExceptionCode::DataError, "Failed to execute 'cmp' on 'IDBFactory': The parameter is not a valid key."_s };

    auto second = scriptValueToIDBKey(execState, secondValue);
    if (!second->isValid())
        return Exception { ExceptionCode::DataError, "Failed to execute 'cmp' on 'IDBFactory': The parameter is not a valid key."_s };

    return first->compare(second.get());
}

void IDBFactory::databases(ScriptExecutionContext& context, IDBDatabasesResponsePromise&& promise)
{
    LOG(IndexedDB, "IDBFactory::databases");

    if (shouldThrowSecurityException(context)) {
        promise.reject(ExceptionCode::SecurityError);
        return;
    }

    ASSERT(context.securityOrigin());

    m_connectionProxy->getAllDatabaseNamesAndVersions(context, [promise = WTFMove(promise)](auto&& result) mutable {
        if (!result) {
            promise.reject(Exception { ExceptionCode::UnknownError });
            return;
        }

        promise.resolve(WTF::map(*result, [](auto&& info) {
            return IDBFactory::DatabaseInfo { WTFMove(info.name), info.version };
        }));
    });
}

void IDBFactory::getAllDatabaseNames(ScriptExecutionContext& context, Function<void(const Vector<String>&)>&& callback)
{
    m_connectionProxy->getAllDatabaseNamesAndVersions(context, [callback = WTFMove(callback)](auto&& result) mutable {
        if (!result) {
            callback({ });
            return;
        }

        callback(WTF::map(*result, [](auto&& info) {
            return WTFMove(info.name);
        }));
    });
}

} // namespace WebCore
