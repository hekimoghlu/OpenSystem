/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 10, 2023.
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
#include "IDBDatabaseNameAndVersionRequest.h"

#include "IDBConnectionProxy.h"
#include "ScriptExecutionContext.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(IDBDatabaseNameAndVersionRequest);

Ref<IDBDatabaseNameAndVersionRequest> IDBDatabaseNameAndVersionRequest::create(ScriptExecutionContext& context, IDBClient::IDBConnectionProxy& connectionProxy, InfoCallback&& callback)
{
    auto result = adoptRef(*new IDBDatabaseNameAndVersionRequest(context, connectionProxy, WTFMove(callback)));
    result->suspendIfNeeded();
    return result;
}

IDBDatabaseNameAndVersionRequest::IDBDatabaseNameAndVersionRequest(ScriptExecutionContext& context, IDBClient::IDBConnectionProxy& connectionProxy, InfoCallback&& callback)
    : IDBActiveDOMObject(&context)
    , m_connectionProxy(connectionProxy)
    , m_resourceIdentifier(connectionProxy)
    , m_callback(WTFMove(callback))
{
    ASSERT(canCurrentThreadAccessThreadLocalData(originThread()));
}

void IDBDatabaseNameAndVersionRequest::complete(std::optional<Vector<IDBDatabaseNameAndVersion>>&& databases)
{
    ASSERT(canCurrentThreadAccessThreadLocalData(originThread()));

    if (auto callback = WTFMove(m_callback))
        callback(WTFMove(databases));
}

bool IDBDatabaseNameAndVersionRequest::virtualHasPendingActivity() const
{
    ASSERT(canCurrentThreadAccessThreadLocalData(originThread()) || Thread::mayBeGCThread());
    return true;
}

void IDBDatabaseNameAndVersionRequest::stop()
{
    ASSERT(canCurrentThreadAccessThreadLocalData(originThread()));
}

} // namespace WebCore
