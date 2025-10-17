/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 26, 2024.
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

#include "IDBDatabaseIdentifier.h"
#include "IDBIndexIdentifier.h"
#include "IDBObjectStoreIdentifier.h"
#include "IDBResourceIdentifier.h"
#include "IndexedDB.h"
#include <optional>
#include <wtf/ArgumentCoder.h>
#include <wtf/Markable.h>

namespace WebCore {

class IDBOpenDBRequest;
class IDBTransaction;

namespace IndexedDB {
enum class IndexRecordType : bool;
}

namespace IDBClient {
class IDBConnectionProxy;
class TransactionOperation;
}

class IDBRequestData {
public:
    explicit IDBRequestData(IDBClient::TransactionOperation&);
    WEBCORE_EXPORT IDBRequestData(const IDBRequestData&);
    IDBRequestData(IDBRequestData&&) = default;
    IDBRequestData& operator=(IDBRequestData&&) = default;

    enum IsolatedCopyTag { IsolatedCopy };
    IDBRequestData(const IDBRequestData&, IsolatedCopyTag);
    WEBCORE_EXPORT IDBRequestData isolatedCopy() const;

    IDBConnectionIdentifier serverConnectionIdentifier() const;
    WEBCORE_EXPORT IDBResourceIdentifier requestIdentifier() const;
    WEBCORE_EXPORT IDBResourceIdentifier transactionIdentifier() const;
    IDBObjectStoreIdentifier objectStoreIdentifier() const;
    std::optional<IDBIndexIdentifier> indexIdentifier() const;
    IndexedDB::IndexRecordType indexRecordType() const;
    IDBResourceIdentifier cursorIdentifier() const;
    uint64_t requestedVersion() const;
    IDBRequestData isolatedCopy();

private:
    friend struct IPC::ArgumentCoder<IDBRequestData, void>;
    WEBCORE_EXPORT IDBRequestData(IDBConnectionIdentifier serverConnectionIdentifier, IDBResourceIdentifier requestIdentifier, IDBResourceIdentifier transactionIdentifier, std::optional<IDBResourceIdentifier>&& cursorIdentifier, std::optional<IDBObjectStoreIdentifier>, std::optional<IDBIndexIdentifier>, IndexedDB::IndexRecordType, uint64_t requestedVersion, IndexedDB::RequestType);
    static void isolatedCopy(const IDBRequestData& source, IDBRequestData& destination);

    IDBConnectionIdentifier m_serverConnectionIdentifier;
    IDBResourceIdentifier m_requestIdentifier;
    IDBResourceIdentifier m_transactionIdentifier;
    std::optional<IDBResourceIdentifier> m_cursorIdentifier;
    Markable<IDBObjectStoreIdentifier> m_objectStoreIdentifier;
    Markable<IDBIndexIdentifier> m_indexIdentifier;
    IndexedDB::IndexRecordType m_indexRecordType { IndexedDB::IndexRecordType::Key };
    uint64_t m_requestedVersion { 0 };

    IndexedDB::RequestType m_requestType { IndexedDB::RequestType::Other };
};

} // namespace WebCore
