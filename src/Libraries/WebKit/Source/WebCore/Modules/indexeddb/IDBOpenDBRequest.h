/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 7, 2024.
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
#include "IDBRequest.h"

namespace WebCore {

class IDBResultData;

class IDBOpenDBRequest final : public IDBRequest {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(IDBOpenDBRequest);
public:
    static Ref<IDBOpenDBRequest> createDeleteRequest(ScriptExecutionContext&, IDBClient::IDBConnectionProxy&, const IDBDatabaseIdentifier&);
    static Ref<IDBOpenDBRequest> createOpenRequest(ScriptExecutionContext&, IDBClient::IDBConnectionProxy&, const IDBDatabaseIdentifier&, uint64_t version);

    virtual ~IDBOpenDBRequest();
    
    const IDBDatabaseIdentifier& databaseIdentifier() const { return m_databaseIdentifier; }
    uint64_t version() const { return m_version; }

    void requestCompleted(const IDBResultData&);
    void requestBlocked(uint64_t oldVersion, uint64_t newVersion);

    void versionChangeTransactionDidFinish();
    void fireSuccessAfterVersionChangeCommit();
    void fireErrorAfterVersionChangeCompletion();

    void setIsContextSuspended(bool);
    bool isContextSuspended() const { return m_isContextSuspended; }

private:
    IDBOpenDBRequest(ScriptExecutionContext&, IDBClient::IDBConnectionProxy&, const IDBDatabaseIdentifier&, uint64_t version, IndexedDB::RequestType);

    void dispatchEvent(Event&) final;

    void cancelForStop() final;

    void onError(const IDBResultData&);
    void onSuccess(const IDBResultData&);
    void onUpgradeNeeded(const IDBResultData&);
    void onDeleteDatabaseSuccess(const IDBResultData&);

    bool isOpenDBRequest() const final { return true; }

    IDBDatabaseIdentifier m_databaseIdentifier;
    uint64_t m_version { 0 };

    bool m_isContextSuspended { false };
    bool m_isBlocked { false };
};

} // namespace WebCore
