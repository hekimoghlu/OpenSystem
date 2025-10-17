/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 26, 2024.
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
#include "IDBResourceIdentifier.h"

#include "IDBConnectionToClient.h"
#include "IDBConnectionToServer.h"
#include "IDBRequest.h"
#include <wtf/MainThread.h>
#include <wtf/text/MakeString.h>

namespace WebCore {

static uint64_t nextClientResourceNumber()
{
    static std::atomic<uint64_t> currentNumber(1);
    return currentNumber += 2;
}

static uint64_t nextServerResourceNumber()
{
    static uint64_t currentNumber = 0;
    return currentNumber += 2;
}

IDBResourceIdentifier::IDBResourceIdentifier() = default;

IDBResourceIdentifier::IDBResourceIdentifier(std::optional<IDBConnectionIdentifier> connectionIdentifier, uint64_t resourceIdentifier)
    : m_idbConnectionIdentifier(connectionIdentifier)
    , m_resourceNumber(resourceIdentifier)
{
}

IDBResourceIdentifier::IDBResourceIdentifier(const IDBClient::IDBConnectionProxy& connectionProxy)
    : m_idbConnectionIdentifier(connectionProxy.serverConnectionIdentifier())
    , m_resourceNumber(nextClientResourceNumber())
{
}

IDBResourceIdentifier::IDBResourceIdentifier(const IDBClient::IDBConnectionProxy& connectionProxy, const IDBRequest& request)
    : m_idbConnectionIdentifier(connectionProxy.serverConnectionIdentifier())
    , m_resourceNumber(request.resourceIdentifier().m_resourceNumber)
{
}

IDBResourceIdentifier::IDBResourceIdentifier(const IDBServer::IDBConnectionToClient& connection)
    : m_idbConnectionIdentifier(connection.identifier())
    , m_resourceNumber(nextServerResourceNumber())
{
}

IDBResourceIdentifier IDBResourceIdentifier::isolatedCopy() const
{
    return IDBResourceIdentifier(m_idbConnectionIdentifier, m_resourceNumber);
}

#if !LOG_DISABLED

String IDBResourceIdentifier::loggingString() const
{
    return makeString('<', m_idbConnectionIdentifier ? m_idbConnectionIdentifier->toUInt64() : 0, ", "_s, m_resourceNumber, '>');
}

#endif

} // namespace WebCore
