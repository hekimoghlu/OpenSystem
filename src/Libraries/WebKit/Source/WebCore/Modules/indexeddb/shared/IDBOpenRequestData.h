/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 9, 2023.
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
#include "IDBResourceIdentifier.h"
#include "IndexedDB.h"
#include <wtf/ArgumentCoder.h>

namespace WebCore {

class IDBOpenDBRequest;

class IDBOpenRequestData {
public:
    IDBOpenRequestData(const IDBClient::IDBConnectionProxy&, const IDBOpenDBRequest&);
    IDBOpenRequestData(const IDBOpenRequestData&) = default;
    IDBOpenRequestData(IDBOpenRequestData&&) = default;
    IDBOpenRequestData& operator=(IDBOpenRequestData&&) = default;
    WEBCORE_EXPORT IDBOpenRequestData isolatedCopy() const;

    IDBConnectionIdentifier serverConnectionIdentifier() const { return m_serverConnectionIdentifier; }
    IDBResourceIdentifier requestIdentifier() const { return m_requestIdentifier; }
    IDBDatabaseIdentifier databaseIdentifier() const { return m_databaseIdentifier; }
    uint64_t requestedVersion() const { return m_requestedVersion; }
    bool isOpenRequest() const { return m_requestType == IndexedDB::RequestType::Open; }
    bool isDeleteRequest() const { return m_requestType == IndexedDB::RequestType::Delete; }

private:
    friend struct IPC::ArgumentCoder<IDBOpenRequestData, void>;
    WEBCORE_EXPORT IDBOpenRequestData(IDBConnectionIdentifier, IDBResourceIdentifier, IDBDatabaseIdentifier&&, uint64_t requestedVersion, IndexedDB::RequestType);

    IDBConnectionIdentifier m_serverConnectionIdentifier;
    IDBResourceIdentifier m_requestIdentifier;
    IDBDatabaseIdentifier m_databaseIdentifier;
    uint64_t m_requestedVersion { 0 };
    IndexedDB::RequestType m_requestType { IndexedDB::RequestType::Open };
};

} // namespace WebCore
