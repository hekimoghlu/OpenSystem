/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 24, 2025.
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
#include "IDBTransactionInfo.h"

#include "IDBTransaction.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/MakeString.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(IDBTransactionInfo);

IDBTransactionInfo::IDBTransactionInfo(const IDBResourceIdentifier& identifier)
    : m_identifier(identifier)
{
}

IDBTransactionInfo IDBTransactionInfo::clientTransaction(const IDBClient::IDBConnectionProxy& connectionProxy, const Vector<String>& objectStores, IDBTransactionMode mode, std::optional<IDBTransactionDurability> durability)
{
    IDBTransactionInfo result((IDBResourceIdentifier(connectionProxy)));
    result.m_objectStores = objectStores;
    result.m_mode = mode;
    if (durability)
        result.m_durability = *durability;

    return result;
}

IDBTransactionInfo IDBTransactionInfo::versionChange(const IDBServer::IDBConnectionToClient& connection, const IDBDatabaseInfo& originalDatabaseInfo, uint64_t newVersion)
{
    IDBTransactionInfo result((IDBResourceIdentifier(connection)));
    result.m_mode = IDBTransactionMode::Versionchange;
    result.m_newVersion = newVersion;
    result.m_originalDatabaseInfo = makeUnique<IDBDatabaseInfo>(originalDatabaseInfo);

    return result;
}

IDBTransactionInfo::IDBTransactionInfo(const IDBTransactionInfo& info)
    : m_identifier(info.identifier())
    , m_mode(info.m_mode)
    , m_durability(info.m_durability)
    , m_newVersion(info.m_newVersion)
    , m_objectStores(info.m_objectStores)
{
    if (info.m_originalDatabaseInfo)
        m_originalDatabaseInfo = makeUnique<IDBDatabaseInfo>(*info.m_originalDatabaseInfo);
}

IDBTransactionInfo::IDBTransactionInfo(const IDBTransactionInfo& that, IsolatedCopyTag)
{
    isolatedCopy(that, *this);
}

IDBTransactionInfo IDBTransactionInfo::isolatedCopy() const
{
    return { *this, IsolatedCopy };
}

void IDBTransactionInfo::isolatedCopy(const IDBTransactionInfo& source, IDBTransactionInfo& destination)
{
    destination.m_identifier = source.m_identifier.isolatedCopy();
    destination.m_mode = source.m_mode;
    destination.m_durability = source.m_durability;
    destination.m_newVersion = source.m_newVersion;

    destination.m_objectStores = source.m_objectStores.map([](auto& objectStore) {
        return objectStore.isolatedCopy();
    });

    if (source.m_originalDatabaseInfo)
        destination.m_originalDatabaseInfo = makeUnique<IDBDatabaseInfo>(*source.m_originalDatabaseInfo, IDBDatabaseInfo::IsolatedCopy);
}

#if !LOG_DISABLED

String IDBTransactionInfo::loggingString() const
{
    String modeString;
    switch (m_mode) {
    case IDBTransactionMode::Readonly:
        modeString = "readonly"_s;
        break;
    case IDBTransactionMode::Readwrite:
        modeString = "readwrite"_s;
        break;
    case IDBTransactionMode::Versionchange:
        modeString = "versionchange"_s;
        break;
    default:
        ASSERT_NOT_REACHED();
    }
    
    return makeString("Transaction: "_s, m_identifier.loggingString(), " mode "_s, modeString, " newVersion "_s, m_newVersion);
}

#endif

} // namespace WebCore
