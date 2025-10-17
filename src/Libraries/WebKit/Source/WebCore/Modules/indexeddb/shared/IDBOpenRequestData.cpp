/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 17, 2023.
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
#include "IDBOpenRequestData.h"

#include "IDBConnectionProxy.h"
#include "IDBDatabaseIdentifier.h"

namespace WebCore {

IDBOpenRequestData::IDBOpenRequestData(const IDBClient::IDBConnectionProxy& connectionProxy, const IDBOpenDBRequest& request)
    : m_serverConnectionIdentifier(connectionProxy.serverConnectionIdentifier())
    , m_requestIdentifier(IDBResourceIdentifier { connectionProxy, request })
    , m_databaseIdentifier(request.databaseIdentifier())
    , m_requestedVersion(request.version())
    , m_requestType(request.requestType())
{
}

IDBOpenRequestData::IDBOpenRequestData(IDBConnectionIdentifier serverConnectionIdentifier, IDBResourceIdentifier requestIdentifier, IDBDatabaseIdentifier&& databaseIdentifier, uint64_t requestedVersion, IndexedDB::RequestType type)
    : m_serverConnectionIdentifier(serverConnectionIdentifier)
    , m_requestIdentifier(requestIdentifier)
    , m_databaseIdentifier(databaseIdentifier)
    , m_requestedVersion(requestedVersion)
    , m_requestType(type)
{
}

IDBOpenRequestData IDBOpenRequestData::isolatedCopy() const
{
    return IDBOpenRequestData {
        m_serverConnectionIdentifier,
        m_requestIdentifier,
        m_databaseIdentifier.isolatedCopy(),
        m_requestedVersion,
        m_requestType
    };
}

} // namespace WebCore
