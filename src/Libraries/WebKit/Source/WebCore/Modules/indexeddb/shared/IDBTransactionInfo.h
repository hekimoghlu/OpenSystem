/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 13, 2025.
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

#include "IDBDatabaseInfo.h"
#include "IDBResourceIdentifier.h"
#include "IDBTransactionDurability.h"
#include "IDBTransactionMode.h"
#include "IndexedDB.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

namespace WebCore {

namespace IDBClient {
class IDBConnectionProxy;
}

namespace IDBServer {
class IDBConnectionToClient;
}

class IDBTransactionInfo {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(IDBTransactionInfo, WEBCORE_EXPORT);
public:
    static IDBTransactionInfo clientTransaction(const IDBClient::IDBConnectionProxy&, const Vector<String>& objectStores, IDBTransactionMode, std::optional<IDBTransactionDurability>);
    static IDBTransactionInfo versionChange(const IDBServer::IDBConnectionToClient&, const IDBDatabaseInfo& originalDatabaseInfo, uint64_t newVersion);

    WEBCORE_EXPORT IDBTransactionInfo(const IDBTransactionInfo&);
    IDBTransactionInfo(IDBTransactionInfo&&) = default;
    IDBTransactionInfo& operator=(IDBTransactionInfo&&) = default;

    enum IsolatedCopyTag { IsolatedCopy };
    IDBTransactionInfo(const IDBTransactionInfo&, IsolatedCopyTag);

    WEBCORE_EXPORT IDBTransactionInfo isolatedCopy() const;

    const IDBResourceIdentifier& identifier() const { return m_identifier; }

    IDBTransactionMode mode() const { return m_mode; }
    IDBTransactionDurability durability() const { return m_durability; }
    uint64_t newVersion() const { return m_newVersion; }

    const Vector<String>& objectStores() const { return m_objectStores; }

    const std::unique_ptr<IDBDatabaseInfo>& originalDatabaseInfo() const { return m_originalDatabaseInfo; }

    IDBTransactionInfo(IDBResourceIdentifier identifier, IDBTransactionMode mode, IDBTransactionDurability durability, uint64_t newVersion, Vector<String>&& objectStores, std::unique_ptr<IDBDatabaseInfo> originalDatabaseInfo)
        : m_identifier(identifier)
        , m_mode(mode)
        , m_durability(durability)
        , m_newVersion(newVersion)
        , m_objectStores(WTFMove(objectStores))
        , m_originalDatabaseInfo(WTFMove(originalDatabaseInfo)) { }

#if !LOG_DISABLED
    String loggingString() const;
#endif

private:
    IDBTransactionInfo(const IDBResourceIdentifier&);

    static void isolatedCopy(const IDBTransactionInfo& source, IDBTransactionInfo& destination);

    IDBResourceIdentifier m_identifier;

    IDBTransactionMode m_mode { IDBTransactionMode::Readonly };
    IDBTransactionDurability m_durability { IDBTransactionDurability::Default };
    uint64_t m_newVersion { 0 };
    Vector<String> m_objectStores;
    std::unique_ptr<IDBDatabaseInfo> m_originalDatabaseInfo;
};

} // namespace WebCore
